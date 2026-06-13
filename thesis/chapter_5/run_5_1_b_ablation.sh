#!/bin/bash
# =============================================================================
# 5.1.2  主動學習超參數敏感度 — 變動 b（每輪查詢間隔），固定 b₀=2.5%
#
#   b = --portion_interval（每個 AL iteration 新增的標註比例）。固定 b₀(portion_start)=2.5%。
#   主結果（Ch4）是 b=2.5%；本腳本讓 b 由外部指定（如 5、10），跑到 ρ≈60% 為止，
#   用來 justify「b=2.5% 最佳（越小越好）」。
#
#   結果隔離（勿與主實驗/其他 b 混）：每個 b 各自一棵獨立樹
#     classification/exp_results/ch5_b_ablation/b_<B>/classification_hard/AL_simclr/...
#   b=2.5% 不用在這裡跑 —— 直接沿用 Ch4 主結果 classification_hard/AL_simclr/。
#
#   初始 ρ=2.5% 步 = 該 seed 的 random 選樣，故沿用 θ² cold-start (2.5,seed) best lr
#   （--coldstart_lr_path 指向真 cold-start 樹）；後續步 sweep + best-val（option A）。
#
#   重跑安全：(b, strategy, seed) JSON 已存在 → 跳過；FORCE=1 強制接續。
#
#   用法（repo 根）：
#     B=5  DEVICE=cuda:0 STRATEGIES="margin coreset cluster_margin" SEEDS="10 24 38 42 57" \
#         ./thesis/chapter_5/run_5_1_b_ablation.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo 根

B=${B:?請指定 b，例如 B=5（每輪查詢間隔 %）}
DEVICE=${DEVICE:-"cuda:0"}
STRATEGIES=${STRATEGIES:-"margin coreset cluster_margin"}
SEEDS=${SEEDS:-"10 24 38 42 57"}
SIMCLR_CKPT=${SIMCLR_CKPT:-"./SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"}
PORTION_START=${PORTION_START:-2.5}     # b₀ 固定 2.5%
# PORTION_END 動態算：讓最後一個 iteration 的 portion >= 60（不論 b 多大）。
#   從 2.5 起每次 +b，到第一個 >=60 的點 last，再設 end=last+b/2（exclusive 上界 → 含 last、不含 last+b）。
#   例：b=5 → last=62.5, end=65.0；b=10 → last=62.5, end=67.5；b=2.5 → last=60, end=61.25。
PORTION_END=${PORTION_END:-$(awk -v s="$PORTION_START" -v b="$B" \
  'BEGIN{last=s; while(last<60) last+=b; printf "%.4g", last + b/2}')}
EXP_PATH="./classification/exp_results/ch5_b_ablation/b_${B}"
COLDSTART_LR_PATH=${COLDSTART_LR_PATH:-"./classification/exp_results"}   # 初始步 lr 查真 cold-start 樹

[ -f "$SIMCLR_CKPT" ] || { echo "!! 找不到 θ² ckpt: $SIMCLR_CKPT"; exit 1; }

echo "############################################################"
echo "# 5.1.2 b ablation   b₀=${PORTION_START}%(固定)   b(interval)=${B}%   →ρ≈60%"
echo "# device=$DEVICE  strategies=[$STRATEGIES]  seeds=[$SEEDS]"
echo "# out=$EXP_PATH/classification_hard/AL_simclr/"
echo "############################################################"

for strat in $STRATEGIES; do
  for seed in $SEEDS; do
    result_json="$EXP_PATH/classification_hard/AL_simclr/${strat}_seed${seed}_bs16.json"
    if [ -f "$result_json" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "!! 已存在，跳過：$result_json  (要強制重跑加 FORCE=1)"
      continue
    fi
    echo "============================================================"
    echo "b=${B}%  strategy=$strat  seed=$seed"
    echo "============================================================"
    python3 ./classification/run_AL.py \
        --task_type hard \
        --AL_strategy "$strat" \
        --pretrained_weights simclr --simclr_path "$SIMCLR_CKPT" \
        --lr_schedule sweep \
        --exp_path "$EXP_PATH" \
        --coldstart_lr_path "$COLDSTART_LR_PATH" \
        --portion_start "$PORTION_START" \
        --portion_end "$PORTION_END" \
        --portion_interval "$B" \
        --seed "$seed" \
        --aug_factor 4 \
        --device "$DEVICE" || true
  done
done

echo "完成 b=${B}%。結果：$EXP_PATH/classification_hard/AL_simclr/"
