#!/bin/bash
# =============================================================================
# 4.5  綜合比較（factor ablation）— Aug(4x) × Init(θ²) × AL(Margin)
#
#   4.2–4.4 是「逐一疊加」策略；4.5 拆解三策略各自的貢獻。兩個 Table（rows ρ=10/30/50）：
#     Table 1（一次只開一個）：Data Aug(4x) | Weight Init(θ²) | AL(Margin) | All Three
#     Table 2（一次只關一個）：w/o Aug      | w/o Init        | w/o AL     | All Three
#
#   7 個 cell 中 3 個直接用主實驗既有資料（不需跑）：
#     aug_only  (aug4,   ImageNet, passive) = 4.2 cold_start_imagenet
#     wo_al     (aug4,   θ²,       passive) = 4.3 cold_start_simclr（best cfg）
#     all_three (aug4,   θ²,       margin)  = 4.4 AL_simclr/margin_*
#   本腳本補剩下 4 個 arm：
#     ARM=init_only  (no_aug, θ²,       passive)  cold-start，只跑 ρ=10/30/50（便宜）
#     ARM=al_only    (no_aug, ImageNet, margin)   AL 軌跡 ρ=2.5→50（interval 2.5）
#     ARM=wo_aug     (no_aug, θ²,       margin)   AL 軌跡 ρ=2.5→50
#     ARM=wo_init    (aug4,   ImageNet, margin)   AL 軌跡 ρ=2.5→50
#
#   結果獨立存放（與主實驗完全隔離，勿混）：
#     classification/exp_results/chapter4_5_ablation/{arm}/classification_hard/...
#
#   慣例（與主實驗對齊）：
#     - cold-start arm：per-portion 下游 lr 網格同 4.3 精簡版（10→4 lr、30→3 lr、50→2 lr），
#       5 seeds × 3 runs；已滿 3 runs 的 (seed,portion,lr) 由 check_existing_results 自動跳過。
#     - AL arm：同 4.4 option A —— lr sweep（3e-5 5e-5 1e-4 3e-4）、每 lr 1 run、
#       best-val model 當選取器、5 seeds。初始 2.5% 在獨立 exp_path 下查不到 cold-start
#       參照 → 一律退回 sweep（4 個 AL cell 一致，含 all_three 以外的三個新 arm）。
#     - AL arm 重跑安全：該 (strategy,seed) 結果 JSON 已存在就跳過；FORCE=1 強制接續。
#
#   AL 代表策略 = margin（user 定案 2026-06-12：所有方法中最好）。要換策略時加
#     STRATEGY=xxx 重跑三個 AL arm 即可（結果檔名含策略名、可並存；init_only 不受影響）。
#
#   用法（repo 根；AL arm 可用 SEEDS 拆卡並行）：
#     ARM=init_only DEVICE=cuda:0 ./thesis/chapter_4/run_4_5_ablation.sh
#     ARM=al_only   DEVICE=cuda:1 ./thesis/chapter_4/run_4_5_ablation.sh
#     ARM=wo_aug    DEVICE=cuda:2 SEEDS="10 24 38" ./thesis/chapter_4/run_4_5_ablation.sh
#     ARM=wo_init   DEVICE=cuda:3 SEEDS="42 57"    ./thesis/chapter_4/run_4_5_ablation.sh
#   檢視兩個 Table（缺 cell 標 NA）：python3 thesis/chapter_4/aggregate_4_5.py
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo 根

: "${ARM:?必須指定 ARM=init_only|al_only|wo_aug|wo_init}"
DEVICE=${DEVICE:-"cuda:0"}
SEEDS=${SEEDS:-"10 24 38 42 57"}
SIMCLR_CKPT=${SIMCLR_CKPT:-"./SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"}
ABL_ROOT="./classification/exp_results/chapter4_5_ablation"
STRATEGY=${STRATEGY:-"margin"}     # 4.5 以 margin 為 AL 代表（4.4 最佳，user 定案 2026-06-12）
PORTION_START=${PORTION_START:-2.5}
PORTION_END=${PORTION_END:-52.5}           # exclusive in np.arange → 跑到 50
PORTION_INTERVAL=${PORTION_INTERVAL:-2.5}

case "$ARM" in
  init_only|al_only|wo_aug|wo_init) ;;
  *) echo "!! ARM 必須是 init_only|al_only|wo_aug|wo_init（給的是 '$ARM'）"; exit 1 ;;
esac

# θ² ckpt 只有用到 simclr init 的 arm 需要
if [ "$ARM" = "init_only" ] || [ "$ARM" = "wo_aug" ]; then
  [ -f "$SIMCLR_CKPT" ] || { echo "!! 找不到 θ² ckpt: $SIMCLR_CKPT"; exit 1; }
fi

# =====================================================================
# ARM=init_only：no_aug + θ² cold-start @ ρ∈{10,30,50}（5 seeds × 3 runs × lr 網格）
#   run_first_iter_simclr.py 需從 classification/ 跑（local imports）。
#   worker pool 與 4_3 相同：seed 最內層派發 → 並行的是不同 seed → 不同 JSON 檔。
# =====================================================================
if [ "$ARM" = "init_only" ]; then
  PORTIONS=${PORTIONS:-"10 30 50"}
  RUNS=${RUNS:-3}
  MAX_PAR=${MAX_PAR:-3}
  cd classification

  down_lrs_for () {            # 同 run_4_3_simclr_finetune.sh 的精簡網格
    case "$1" in
      2.5|5|10)           echo "3e-5 5e-5 1e-4 3e-4" ;;
      15|20|25|30|35|40)  echo "5e-5 1e-4 5e-4" ;;
      *)                  echo "1e-4 5e-4" ;;   # 45–90
    esac
  }

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

  echo "ARM=init_only (no_aug + θ² cold-start)  PORTIONS=[$PORTIONS]  SEEDS=[$SEEDS]  MAX_PAR=$MAX_PAR  DEVICE=$DEVICE"
  for p in $PORTIONS; do
    for dlr in $(down_lrs_for "$p"); do
      for r in $(seq 1 "$RUNS"); do
        for s in $SEEDS; do
          wait_slot
          echo "[init_only] ρ=$p lr=$dlr seed=$s run=$r (par≤$MAX_PAR)"
          ( python3 ./run_first_iter_simclr.py \
              --task_type hard --pretrained_weights simclr --simclr_init imagenet \
              --simclr_lr 0.0002 --simclr_bs 256 --simclr_ep 500 \
              --portion "$p" --seed "$s" --lr "$dlr" --no_data_aug \
              --exp_path ./exp_results/chapter4_5_ablation/init_only \
              --device "$DEVICE" || true ) &
          pids+=($!)
        done
      done
    done
  done
  wait
  echo "init_only 完成。檢視：python3 thesis/chapter_4/aggregate_4_5.py"
  exit 0
fi

# =====================================================================
# AL arms：cluster_margin 軌跡 ρ=2.5→50（同 4.4 option A，差別只在 aug / init）
# =====================================================================
case "$ARM" in
  al_only)  PRETRAIN=imagenet; AUG_FLAGS="--no_data_aug";                            AL_DIR=AL_imagenet ;;
  wo_aug)   PRETRAIN=simclr;   AUG_FLAGS="--no_data_aug";                            AL_DIR=AL_simclr ;;
  wo_init)  PRETRAIN=imagenet; AUG_FLAGS="--aug_factor 4";                           AL_DIR=AL_imagenet ;;
esac
SIMCLR_FLAGS=""
[ "$PRETRAIN" = "simclr" ] && SIMCLR_FLAGS="--simclr_path $SIMCLR_CKPT"
EXP_PATH="$ABL_ROOT/$ARM"

echo "ARM=$ARM (init=$PRETRAIN, aug='$AUG_FLAGS', strategy=$STRATEGY)  SEEDS=[$SEEDS]  DEVICE=$DEVICE"
for seed in $SEEDS; do
  result_json="$EXP_PATH/classification_hard/$AL_DIR/${STRATEGY}_seed${seed}_bs16.json"
  if [ -f "$result_json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "!! existing file already exists, skip run: $result_json"
    echo "   (不論是否跑完所有 portion 都跳過；要強制重跑/接續請加 FORCE=1)"
    continue
  fi
  echo "============================================================"
  echo "4.5 ABLATION: arm=$ARM  strategy=$STRATEGY  seed=$seed"
  echo "============================================================"
  python3 ./classification/run_AL.py \
      --task_type hard \
      --AL_strategy "$STRATEGY" \
      --pretrained_weights "$PRETRAIN" $SIMCLR_FLAGS \
      --lr_schedule sweep \
      --exp_path "$EXP_PATH" \
      --portion_start "$PORTION_START" \
      --portion_end "$PORTION_END" \
      --portion_interval "$PORTION_INTERVAL" \
      --seed "$seed" \
      $AUG_FLAGS \
      --device "$DEVICE" || true
done

echo "$ARM 完成。檢視：python3 thesis/chapter_4/aggregate_4_5.py"
