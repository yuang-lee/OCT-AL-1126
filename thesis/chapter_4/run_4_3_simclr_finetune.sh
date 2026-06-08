#!/bin/bash
# =============================================================================
# 4.3  SimCLR finetune meta script —— 一支搞定 θ¹ / θ²
#
#   θ¹ 與 θ² 用「完全相同的 SimCLR recipe」(lr0.0002/bs256/ep500)，只差 init
#   → 受控比較。本腳本：指定 init、並行數、portion 範圍即可。
#
#   必填：
#     INIT=theta1|theta2     theta1=random→SimCLR（寫 cold_start_simclr_randinit）
#                            theta2=ImageNet→SimCLR（寫 cold_start_simclr）
#     PORTIONS="..."         要跑的 portion（空白分隔，如 "20 30 40"）
#     DEVICE=cuda:N          注意 cuda:N ≠ 實體 GPU，見 repo 根 gpu_map.md
#   選填：
#     MAX_PAR  一卡並行幾個 process（預設 3，小 portion 一個吃不滿 GPU）
#     SEEDS    預設 "10 24 38 42 57"；ρ=100 自動只用單一 seed（與 seed 無關）
#     RUNS     每 (seed,lr) 重複次數（預設 3）
#     SIMCLR_LR/SIMCLR_BS/SIMCLR_EP  預設 0.0002 / 256 / 500
#
#   範例（見 README A / C）：
#     INIT=theta1 PORTIONS="2.5 5 10 20 30 40 50" DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
#     INIT=theta1 PORTIONS="80 90"  MAX_PAR=3 DEVICE=cuda:3 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
#     INIT=theta2 PORTIONS="5 20 30 40 50 60 70 80 90" DEVICE=cuda:5 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
#
#   驗證：python3 thesis/chapter_4/aggregate_results.py --theta1
#         python3 thesis/chapter_4/plot_portion_curve.py
#   重跑安全：已完成的 (seed,portion,lr) 自動跳過（會印 Experiment already completed，正常）。
# =============================================================================
set -e
cd "$(dirname "$0")/../../classification"

: "${INIT:?必須指定 INIT=theta1 或 theta2}"
: "${PORTIONS:?必須指定 PORTIONS，例如 PORTIONS=\"20 30 40\"}"
case "$INIT" in
  theta1|random)   simclr_init=random;   prefix=resnet18_random_simclr ;;
  theta2|imagenet) simclr_init=imagenet; prefix=resnet18_simclr ;;
  *) echo "!! INIT 必須是 theta1 或 theta2（給的是 '$INIT'）"; exit 1 ;;
esac

DEVICE=${DEVICE:-cuda:0}
# 並行數 MAX_PAR：未指定時依 GPU 記憶體自動決定
#   ≥40G（A6000 / RTX 6000 Ada 等 48G 大卡）→ 5；其餘（如 3090 24G）→ 3
if [ -z "${MAX_PAR:-}" ]; then
  read -r MAX_PAR _gpu < <(python3 -c "import torch; i=int('${DEVICE#cuda:}'); p=torch.cuda.get_device_properties(i); print(5 if p.total_memory/1e9>40 else 3, p.name)" 2>/dev/null || echo "3 unknown")
  echo "auto MAX_PAR=$MAX_PAR  (GPU: $_gpu)"
fi
SEEDS=${SEEDS:-"10 24 38 42 57"}          # ρ<70 用全部 5 個
SEEDS_HIGH=${SEEDS_HIGH:-"10 24 42"}      # ρ≥70（<100）只用 3 個（訓練慢、子集變異小）
RUNS=${RUNS:-3}
SIMCLR_LR=$(python3 -c "print(str(float('${SIMCLR_LR:-0.0002}')))")   # 2e-4→0.0002 正規化
BS=${SIMCLR_BS:-256}; EP=${SIMCLR_EP:-500}

ckpt="../SSL/simclr/ckpt/${prefix}_lr${SIMCLR_LR}_bs${BS}_ep${EP}.pkl"
[ -f "$ckpt" ] || { echo "!! 找不到 ckpt: $ckpt （θ¹ 需先跑 run_4_3_theta1_pretrain.sh）"; exit 1; }
echo "INIT=$INIT ($simclr_init)  PORTIONS=[$PORTIONS]  MAX_PAR=$MAX_PAR  DEVICE=$DEVICE  ckpt=$(basename "$ckpt")"

# 每個 portion 對應的下游 lr 掃描網格（user 精簡版，2026-06-08）。
# 依 θ_ImageNet/θ² 最佳-lr 趨勢精簡：預訓練 init 最佳 lr 落在 {5e-5…5e-4}，用不到 7e-6/1e-5。
# ⚠️ 取捨：mid/high 刻意不含 3e-4（它其實是 ρ=20/30/50/60/70 的實測最佳），為求快而省，
#    可能讓 mid portion 系統性低 ~1–2%（user 已知並接受）。best-lr 慣例下選次佳的 1e-4/5e-4。
down_lrs_for () {
  case "$1" in
    2.5|5|10)           echo "3e-5 5e-5 1e-4 3e-4" ;;
    15|20|25|30|35|40)  echo "5e-5 1e-4 5e-4" ;;
    45|50|55|60|65|70)  echo "1e-4 5e-4" ;;
    100)                echo "1e-4 5e-4 7e-4" ;;
    *)                  echo "1e-4 5e-4" ;;   # 75,80,90
  esac
}

# ---- worker pool：最多 MAX_PAR 個並行；Ctrl-C 清乾淨 ----
# 注意：迴圈變數用 _pid（local），絕不可用 p/s/r/dlr，否則會覆寫外層 portion 等變數！
pids=()
cleanup () { echo "中斷，清理子程序..."; local _pid; for _pid in "${pids[@]}"; do kill -9 "$_pid" 2>/dev/null; done; wait 2>/dev/null; exit 1; }
trap cleanup SIGINT SIGTERM
wait_slot () {
  local alive _pid
  while :; do
    alive=(); for _pid in "${pids[@]}"; do ps -p "$_pid" >/dev/null 2>&1 && alive+=("$_pid"); done
    pids=("${alive[@]}"); [ "${#pids[@]}" -lt "$MAX_PAR" ] && return; sleep 1
  done
}

# 派發順序 p→lr→run→seed（seed 最內）：同時並行的是不同 seed → 不同結果檔 → 避免 JSON 互相覆蓋
for p in $PORTIONS; do
  # ρ=100 全集與 seed 無關 → 單 seed；ρ≥70 訓練慢、子集變異小 → 3 seeds；其餘 5 seeds
  if   [ "$p" = "100" ];               then seeds="42"
  elif awk "BEGIN{exit !($p>=70)}";    then seeds="$SEEDS_HIGH"
  else                                      seeds="$SEEDS"; fi
  for dlr in $(down_lrs_for "$p"); do
    for r in $(seq 1 "$RUNS"); do
      for s in $seeds; do
        wait_slot
        echo "[$INIT] ρ=$p lr=$dlr seed=$s run=$r (par≤$MAX_PAR)"
        ( python3 ./run_first_iter_simclr.py \
            --task_type hard --pretrained_weights simclr --simclr_init "$simclr_init" \
            --simclr_lr "$SIMCLR_LR" --simclr_bs "$BS" --simclr_ep "$EP" \
            --portion "$p" --seed "$s" --lr "$dlr" --aug_factor 4 --device "$DEVICE" || true ) &
        pids+=($!)
      done
    done
  done
done
wait
echo "完成。驗證：python3 thesis/chapter_4/aggregate_results.py --theta1   /   plot_portion_curve.py"
