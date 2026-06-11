#!/usr/bin/env bash
# 學習曲線（seed 10）— Random / ImageNet init，portion 10/30（ρ=100 不畫）。
# 用途：與 seed 42 各一條，之後在 learning curve 上畫 mean±std 陰影區間。
# 關鍵：每個 (init, portion) 用「seed10 在 portion 曲線上的 best lr」（mean-of-runs 選出）。
# 用法：DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s10.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:8}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  pretrained  portion  lr
  echo "=== $1  ρ=$3%  lr=$4  (seed 10) ==="
  python3 classification/run_first_iter.py --task_type hard --pretrained_weights "$2" \
    --portion "$3" --seed 10 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s10.json"
}

# Random Init（seed10 best lr）
run random   random   10  3e-4
run random   random   30  1e-4

# ImageNet Init（seed10 best lr）
run imagenet imagenet 10  3e-4
run imagenet imagenet 30  1e-4

echo "✓ baselines (seed10) done -> $OUT/{random,imagenet}_p{10,30}_s10.json"
