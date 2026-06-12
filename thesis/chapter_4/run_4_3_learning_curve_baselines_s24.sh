#!/usr/bin/env bash
# 學習曲線（seed 24）— Random / ImageNet init，portion 10/30（ρ=100 不畫）。
# 用途：與 seed 10/42 合計 3 條，learning curve 上畫 mean±std 陰影。
# lr = 各 (init, portion) 在 seed24 cold-start sweep 的 best lr（mean-of-runs 選出）。
# 用法：DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s24.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:7}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  pretrained  portion  lr
  echo "=== $1  ρ=$3%  lr=$4  (seed 24) ==="
  python3 classification/run_first_iter.py --task_type hard --pretrained_weights "$2" \
    --portion "$3" --seed 24 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s24.json"
}

# Random Init（seed24 best lr）
run random   random   10  1e-5
run random   random   30  3e-4

# ImageNet Init（seed24 best lr）
run imagenet imagenet 10  3e-4
run imagenet imagenet 30  3e-4

echo "✓ baselines (seed24) done -> $OUT/{random,imagenet}_p{10,30}_s24.json"
