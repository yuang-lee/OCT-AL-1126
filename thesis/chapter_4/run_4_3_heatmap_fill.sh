#!/bin/bash
# =============================================================================
# 4.3  補完 θ²_SimCLR (ImageNet→SimCLR) 的 bs×ep 下游熱力圖
#      目標：填滿 ρ=100% 與 ρ=10% 兩張 grid，驗證「bs↑ / epoch↑ → 下游↑」單調性。
#
#      ★ 全部 30 個 SimCLR checkpoint (lr0.0002, 各 bs×ep) 已存在於
#        SSL/simclr/ckpt/ → 本步驟「不需」重新 pretrain，純下游 finetune。
#      ★ 薄包裝：實際呼叫既有的 classification/exp/weights_init/scripts/simclr_meta.sh
#
#      缺口（由 thesis/chapter_4/aggregate_results.py --heatmap 算出，2026-06-08）：
#        ρ=100: bs{16,32}×ep{100,200,500}  與  bs{128,256}×ep{10,20,50}   (12 格)
#        ρ=10 : bs{128}×ep{20,50,100,200,500}                              (5 格)
#
# 用法（每個區塊可丟不同 GPU 並行）：
#   DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_heatmap_fill.sh rho100
#   DEVICE=cuda:1 ./thesis/chapter_4/run_4_3_heatmap_fill.sh rho10
#   DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_heatmap_fill.sh all
# 跑完用 aggregate_results.py --heatmap 確認沒有 NA 殘留。
# =============================================================================
set -e

DEVICE=${DEVICE:-"cuda:0"}
LRS=${LRS:-"5e-5 1e-4 3e-4"}      # 下游 lr 掃描，best-lr 由 aggregate_results.py 挑
RUNS=${RUNS:-3}                    # 每格訓練次數
SIMCLR_LR=${SIMCLR_LR:-"0.0002"}  # 熱力圖固定 SimCLR lr=2e-4，只變 bs/ep
WHAT=${1:-all}

cd "$(dirname "$0")/../../classification"   # simclr_meta.sh 的相對路徑以 classification/ 為基準
META="./exp/weights_init/scripts/simclr_meta.sh"

run_block () {  # $1=BSS  $2=EPS  $3=PORTIONS  $4=SEEDS
  SIMCLR_BSS="$1" SIMCLR_EPS="$2" PORTIONS="$3" SEEDS="$4" \
  SIMCLR_LR="$SIMCLR_LR" LRS="$LRS" RUNS="$RUNS" MAX_RUN="$RUNS" \
  AUGS="aug4" PRETRAINED="simclr" DEVICE="$DEVICE" \
  bash "$META"
}

if [ "$WHAT" = "rho100" ] || [ "$WHAT" = "all" ]; then
  echo "### ρ=100 heatmap fill (seed42, 單 seed 與現有論文熱力圖一致) ###"
  run_block "16 32"   "100 200 500" "100" "42"   # 左上角：小 bs 高 epoch
  run_block "128 256" "10 20 50"    "100" "42"   # 右下角：大 bs 低 epoch
fi

if [ "$WHAT" = "rho10" ] || [ "$WHAT" = "all" ]; then
  echo "### ρ=10 heatmap fill (5 seeds，與該 portion 其他格一致) ###"
  run_block "128" "20 50 100 200 500" "10" "10 24 38 42 57"   # 補 bs128 整列
fi

if [ "$WHAT" = "rho30" ] || [ "$WHAT" = "all" ]; then
  # ρ=30 熱力圖網格目前 0/30 全空 → 整張都要跑。SEEDS 預設 42（與 ρ=100 單 seed 一致，最便宜）；
  # 若要做 10/30/100 的 portion×bs 交互分析、需多 seed，改 SEEDS="10 24 38 42 57"。
  echo "### ρ=30 heatmap fill (整張 5bs×6ep；SEEDS=${SEEDS_30:-42}) ###"
  run_block "16 32 64 128 256" "10 20 50 100 200 500" "30" "${SEEDS_30:-42}"
fi

echo "完成。檢查覆蓋：python3 thesis/chapter_4/aggregate_results.py --heatmap"
echo "畫 PNG：cd classification/exp/weights_init && python3 plot_simclr_heatmap.py --portion <10|30|100>"
