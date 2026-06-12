#!/bin/bash
# =============================================================================
# 5.1  主動學習超參數敏感度 — 變動 b₀（初始隨機標註比例）
#
#   b₀ = --portion_start（初始 random labeled pool 大小）。b = interval（default 2.5%）。
#   主結果（Ch4）固定 b₀=2.5%；本腳本讓 b₀ 由外部指定，跑 AL 軌跡到 ρ=60% 為止，
#   用來 justify 「b₀=2.5% 已足夠 / 最佳」。
#
#   結果隔離（重要，勿與主實驗混）：每個 b₀ 各自一棵獨立樹
#     classification/exp_results/ch5_b0_ablation/b0_<B0>/classification_hard/AL_simclr/...
#   →（a）與 default 的 classification_hard/AL_simclr/ 完全分開；
#     （b）不同 b₀ 因路徑不同也不會互相覆蓋（檔名都是 {strat}_seed{seed}_bs16.json）。
#
#   重跑安全：若該 (b₀, strategy, seed) 的 JSON 已存在 → 自動跳過（避免重複 append）；
#            要強制接續/覆蓋加 FORCE=1。
#
#   協定：與 4.4 完全相同（option A：每 portion sweep lr、val 最低的 model 當選取器；
#         std 取自多個 seed），θ² best ckpt 初始化。
#   初始 b₀ 步的 lr：初始池 = 該 seed 的 random 選樣（同 seed→同子集），故直接沿用
#     θ² cold-start 在該 (b₀, seed) 的 best lr（免重掃）。靠 --coldstart_lr_path 指向
#     真正的 cold-start 樹（結果仍寫到隔離的 EXP_PATH）。後續 ρ>b₀ 才 sweep。
#
#   用法（repo 根）：
#     B0=5  DEVICE=cuda:0 STRATEGIES="margin coreset badge" SEEDS="10 24 38 42 57" \
#         ./thesis/chapter_5/run_5_1_b0_ablation.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../.."   # repo 根（run_AL.py 從這裡跑）

B0=${B0:?請指定 b₀，例如 B0=5（初始隨機標註比例 %）}
DEVICE=${DEVICE:-"cuda:0"}
STRATEGIES=${STRATEGIES:-"margin coreset badge"}
SEEDS=${SEEDS:-"10 24 38 42 57"}
SIMCLR_CKPT=${SIMCLR_CKPT:-"./SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"}
PORTION_END=${PORTION_END:-62.5}        # exclusive in np.arange → 跑到 60
PORTION_INTERVAL=${PORTION_INTERVAL:-2.5}   # b（每輪間隔），default 2.5%

# 每個 b₀ 一棵獨立結果樹（與 default、與其他 b₀ 完全隔離）
EXP_PATH="./classification/exp_results/ch5_b0_ablation/b0_${B0}"
# 初始 b₀ 步查 cold-start best lr 用「真正的」結果樹（結果仍寫到上面隔離的 EXP_PATH）
COLDSTART_LR_PATH=${COLDSTART_LR_PATH:-"./classification/exp_results"}

[ -f "$SIMCLR_CKPT" ] || { echo "!! 找不到 θ² ckpt: $SIMCLR_CKPT"; exit 1; }

echo "############################################################"
echo "# 5.1 b₀ ablation   b₀(portion_start)=${B0}%   b(interval)=${PORTION_INTERVAL}%   →ρ=60%"
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
    echo "b₀=${B0}%  strategy=$strat  seed=$seed"
    echo "============================================================"
    python3 ./classification/run_AL.py \
        --task_type hard \
        --AL_strategy "$strat" \
        --pretrained_weights simclr --simclr_path "$SIMCLR_CKPT" \
        --lr_schedule sweep \
        --exp_path "$EXP_PATH" \
        --coldstart_lr_path "$COLDSTART_LR_PATH" \
        --portion_start "$B0" \
        --portion_end "$PORTION_END" \
        --portion_interval "$PORTION_INTERVAL" \
        --seed "$seed" \
        --aug_factor 4 \
        --device "$DEVICE" || true
  done
done

echo "完成 b₀=${B0}%。結果：$EXP_PATH/classification_hard/AL_simclr/"
echo "labeled ids：$EXP_PATH/classification_hard/AL_simclr/labeled_ids/"
