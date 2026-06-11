#!/usr/bin/env bash
# 學習曲線 ρ=100（seed 42）— 四種 init 各補跑 2 個獨立 run（run2 / run3）。
# 為何用 runs 而非 seeds：ρ=100 全集被選入、與 seed 無關（random.sample(全集)=全集）
#   → 變異不能來自 seed，改用「同 seed 多次獨立訓練」（model init / shuffle / GPU 非決定性，
#     程式無 torch.manual_seed，故各 run 真的獨立）。與 4.3 ρ=100 慣例一致。
# 與既有 run1（{init}_p100_s42.json，來自 seed42 腳本）合計 3 runs → 畫圖時對 runs 取 mean±std。
# lr = 各 init 在 seed42 ρ=100 的 best lr（與 run1 同）。
# 用法：DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_p100_runs.sh
set -e
cd "$(dirname "$0")/../.."          # repo 根
DEVICE=${DEVICE:-cuda:8}
OUT=thesis/chapter_4/learning_curves
mkdir -p "$OUT"

run_base() {    # name  pretrained  lr  runidx
  echo "=== $1  ρ=100%  lr=$3  run$4  (seed 42) ==="
  python3 classification/run_first_iter.py --task_type hard --pretrained_weights "$2" \
    --portion 100 --seed 42 --lr "$3" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p100_s42_run$4.json"
}

run_simclr() {  # name  simclr_init  lr  runidx
  echo "=== $1  ρ=100%  lr=$3  run$4  (seed 42) ==="
  python3 classification/run_first_iter_simclr.py --task_type hard --pretrained_weights simclr \
    --simclr_init "$2" --simclr_lr 0.0002 --simclr_bs 256 --simclr_ep 500 \
    --portion 100 --seed 42 --lr "$3" --epoch 20 --aug_factor 4 --device "$DEVICE" \
    --save_history "$OUT/$1_p100_s42_run$4.json"
}

# 兩個額外 run（run2、run3）；run1 = 既有 {init}_p100_s42.json
for R in 2 3; do
  run_base   random   random   7e-4 "$R"   # θ_rand     seed42 ρ=100 best lr
  run_base   imagenet imagenet 1e-4 "$R"   # θ_ImageNet
  run_simclr simclr1  random   1e-4 "$R"   # θ¹ (random→SimCLR)
  run_simclr simclr2  imagenet 1e-4 "$R"   # θ² (ImageNet→SimCLR)
done

echo "✓ p100 extra runs done -> $OUT/{random,imagenet,simclr1,simclr2}_p100_s42_run{2,3}.json"
