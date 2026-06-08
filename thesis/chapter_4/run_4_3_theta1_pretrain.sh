#!/bin/bash
# =============================================================================
# 4.3  θ¹_SimCLR (random→SimCLR) — Step 1/3：預訓練
#
#   從 random init 出發做 SimCLR（run.py -a resnet18_random）。
#   固定 bs256/ep500（不做 bs×ep ablation，直接用最好的設定）。
#   ★ lr 不照搬 θ² 的 2e-4：random init 的最佳 lr 與 imagenet init 不同，
#     故掃 {1e-4, 2e-4, 4e-4}，下一步再挑最佳，避免把 θ¹ 調差、使 θ²>θ¹ 贏得不乾淨。
#
#   存檔（run.py 自動依 arch 命名，不會撞到 θ²）：
#     SSL/simclr/ckpt/resnet18_random_simclr_lr{lr}_bs256_ep500.pkl
#     SSL/simclr/json/resnet18_random_simclr_lr{lr}_bs256_ep500.json
#
#   用法（從 repo 根執行）：
#     DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
#   三個 lr 可分丟不同 GPU：
#     DEVICE=cuda:0 SIMCLR_LRS="1e-4" ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
#     DEVICE=cuda:1 SIMCLR_LRS="2e-4" ...
#     DEVICE=cuda:2 SIMCLR_LRS="4e-4" ...
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo 根（run.py 的相對路徑以此為基準）

DEVICE=${DEVICE:-"cuda:0"}
SIMCLR_LRS=${SIMCLR_LRS:-"1e-4 2e-4 4e-4"}
BS=${BS:-256}
EP=${EP:-500}
DATA=${DATA:-"./ds/classification/seven_class/train"}

for lr in $SIMCLR_LRS; do
  echo "============================================================"
  echo "θ¹ SimCLR pretrain (random init):  lr=$lr  bs=$BS  ep=$EP  dev=$DEVICE"
  echo "============================================================"
  python3 ./SSL/simclr/run.py \
      --arch resnet18_random \
      --lr "$lr" \
      --batch-size "$BS" \
      --epochs "$EP" \
      -data "$DATA" \
      --device "$DEVICE"
done

echo ""
echo "完成。產出 resnet18_random_simclr_lr{...}_bs${BS}_ep${EP}.pkl"
echo "下一步：跑 run_4_3_theta1_pick_lr.sh 在 100% data 上各 finetune，挑最佳 lr。"
