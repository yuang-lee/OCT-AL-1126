#!/usr/bin/env bash
# 學習曲線（seed 38）— SimCLR_1(θ¹: random→SimCLR) / SimCLR_2(θ²: ImageNet→SimCLR)，portion 10/30。
# 與 seed 10/24/42/57 合計 5 seeds → learning curve 畫 mean±std 陰影。
# lr = 各 (init, portion) 在 seed38 cold-start sweep 的 best lr（mean-of-runs 選出）。
# SimCLR ckpt 固定 lr0.0002/bs256/ep500（runner 由 --simclr_init 自建路徑）。
# 用法：DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_simclr_s38.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:7}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run() {  # name  simclr_init  portion  lr
  echo "=== $1  ρ=$3%  lr=$4  (seed 38) ==="
  python3 classification/run_first_iter_simclr.py --task_type hard --pretrained_weights simclr \
    --simclr_init "$2" --simclr_lr 0.0002 --simclr_bs 256 --simclr_ep 500 \
    --portion "$3" --seed 38 --lr "$4" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p$3_s38.json"
}

# SimCLR_1（θ¹: random→SimCLR；seed38 best lr）
run simclr1 random   10  3e-5
run simclr1 random   30  1e-4

# SimCLR_2（θ²: ImageNet→SimCLR；seed38 best lr）
run simclr2 imagenet 10  1e-4
run simclr2 imagenet 30  1e-4

echo "✓ simclr (seed38) done -> $OUT/{simclr1,simclr2}_p{10,30}_s38.json"
