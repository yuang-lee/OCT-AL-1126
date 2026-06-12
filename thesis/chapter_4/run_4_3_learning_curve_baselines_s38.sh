#!/usr/bin/env bash
# 學習曲線（seed 38）— Random / ImageNet init，portion 10/30（ρ=100 不畫）。
# 與 seed 10/24/42/57 合計 5 seeds → learning curve 畫 mean±std 陰影。
# lr = 各 (init, portion) 在 seed38 cold-start sweep 的 best lr（mean-of-runs 選出）。
# 用法：DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s38.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:0}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  pretrained  portion  lr
  echo "=== $1  ρ=$3%  lr=$4  (seed 38) ==="
  python3 classification/run_first_iter.py --task_type hard --pretrained_weights "$2" \
    --portion "$3" --seed 38 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s38.json"
}

# Random Init（seed38 best lr）
run random   random   10  7e-6
run random   random   30  5e-4

# ImageNet Init（seed38 best lr）
run imagenet imagenet 10  3e-4
run imagenet imagenet 30  3e-4

echo "✓ baselines (seed38) done -> $OUT/{random,imagenet}_p{10,30}_s38.json"
