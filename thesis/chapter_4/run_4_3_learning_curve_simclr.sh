#!/usr/bin/env bash
# 學習曲線（seed 42）— SimCLR_1(θ¹: random→SimCLR) / SimCLR_2(θ²: ImageNet→SimCLR)，portion 10/30/100。
# 關鍵：每個 (init, portion) 用「seed42 在 portion 曲線上的 best lr」，
#       這樣 learning curve 收斂點才會對上 weight-init portion 圖上 report 的數字。
# SimCLR ckpt 固定 lr0.0002/bs256/ep500（runner 由 --simclr_init 自建路徑）。
# 用法：DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_simclr.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:8}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  simclr_init  portion  lr
  echo "=== $1  ρ=$3%  lr=$4 ==="
  python3 classification/run_first_iter_simclr.py --task_type hard --pretrained_weights simclr \
    --simclr_init "$2" --simclr_lr 0.0002 --simclr_bs 256 --simclr_ep 500 \
    --portion "$3" --seed 42 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s42.json"
}

# SimCLR_1（θ¹: random→SimCLR；seed42 best lr）
run simclr1 random   10  5e-5
run simclr1 random   30  1e-4
run simclr1 random   100 1e-4

# SimCLR_2（θ²: ImageNet→SimCLR；seed42 best lr）
run simclr2 imagenet 10  1e-4
run simclr2 imagenet 30  5e-5
run simclr2 imagenet 100 1e-4

echo "✓ simclr learning curves done -> $OUT/{simclr1,simclr2}_p{10,30,100}_s42.json"
