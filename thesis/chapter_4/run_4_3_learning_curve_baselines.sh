#!/usr/bin/env bash
# 學習曲線（seed 42）— Random / ImageNet init，portion 10/30/100。
# 關鍵：每個 (init, portion) 用「seed42 在 portion 曲線上的 best lr」，
#       這樣 learning curve 收斂點才會對上 weight-init portion 圖上 report 的數字。
# 用法：DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_baselines.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:8}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  pretrained  portion  lr
  echo "=== $1  ρ=$3%  lr=$4 ==="
  python3 classification/run_first_iter.py --task_type hard --pretrained_weights "$2" \
    --portion "$3" --seed 42 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s42.json"
}

# Random Init（seed42 best lr）
run random   random   10  1e-5
run random   random   30  3e-4
run random   random   100 7e-4

# ImageNet Init（seed42 best lr）
run imagenet imagenet 10  1e-4
run imagenet imagenet 30  3e-4
run imagenet imagenet 100 1e-4

echo "✓ baselines learning curves done -> $OUT/{random,imagenet}_p{10,30,100}_s42.json"
