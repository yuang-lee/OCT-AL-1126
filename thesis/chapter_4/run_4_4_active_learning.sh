#!/bin/bash
# =============================================================================
# 4.4  主動學習 — 5 策略 × 5 initial seeds（新協定）
#
#   協定：每個 seed = 一條 AL 軌跡。每 portion 掃多個下游 lr（lr_grid_for），
#         用 val loss 最低的 model 當「選取器」去選下一批（option A）；
#         std 取自 5 seeds。random/passive baseline 不另跑（用 θ² cold-start 曲線）。
#
#   統一用 θ² best checkpoint 初始化（與 4.3 一致）：
#     SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl
#
#   結果：classification/exp_results/classification_hard/AL_simclr/{strategy}_seed{seed}_bs16.json
#   每 portion 選到的 labeled id 另存：AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json
#
#   用法（repo 根）：
#     DEVICE=cuda:0 ./thesis/chapter_4/run_4_4_active_learning.sh
#   只跑部分策略/seeds（可分卡並行）：
#     DEVICE=cuda:0 STRATEGIES="entropy margin" SEEDS="42 10" ./thesis/chapter_4/run_4_4_active_learning.sh
#   重跑安全：若 {strategy}_seed{seed}_bs16.json 已存在（不論跑完與否）會自動跳過；
#   要強制重跑（接續/覆蓋既有 JSON）加 FORCE=1。
#   檢視：python3 thesis/chapter_4/aggregate_results.py    （AL 區塊）
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo 根（run_AL.py 從這裡跑）

DEVICE=${DEVICE:-"cuda:0"}
STRATEGIES=${STRATEGIES:-"random conf entropy margin coreset badge"}
SEEDS=${SEEDS:-"10 24 38 42 57"}
SIMCLR_CKPT=${SIMCLR_CKPT:-"./SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"}
EXP_PATH=${EXP_PATH:-"./classification/exp_results"}
PORTION_START=${PORTION_START:-2.5}   # 2.5% 起始（初始 random labeled pool）
PORTION_END=${PORTION_END:-62.5}      # exclusive in np.arange → 跑到 60
PORTION_INTERVAL=${PORTION_INTERVAL:-2.5}

[ -f "$SIMCLR_CKPT" ] || { echo "!! 找不到 θ² ckpt: $SIMCLR_CKPT"; exit 1; }

for strat in $STRATEGIES; do
  for seed in $SEEDS; do
    result_json="$EXP_PATH/classification_hard/AL_simclr/${strat}_seed${seed}_bs16.json"
    if [ -f "$result_json" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "!! existing file already exists, skip run: $result_json"
      echo "   (不論是否跑完所有 portion 都跳過；要強制重跑請加 FORCE=1)"
      continue
    fi
    echo "============================================================"
    echo "AL: strategy=$strat  seed=$seed  (sweep lr, best-val model selects)"
    echo "============================================================"
    python3 ./classification/run_AL.py \
        --task_type hard \
        --AL_strategy "$strat" \
        --pretrained_weights simclr --simclr_path "$SIMCLR_CKPT" \
        --lr_schedule sweep \
        --exp_path "$EXP_PATH" \
        --portion_start "$PORTION_START" \
        --portion_end "$PORTION_END" \
        --portion_interval "$PORTION_INTERVAL" \
        --seed "$seed" \
        --aug_factor 4 \
        --device "$DEVICE" || true
  done
done

echo "完成。檢視：python3 thesis/chapter_4/aggregate_results.py"
echo "labeled ids：classification/exp_results/classification_hard/AL_simclr/labeled_ids/"
