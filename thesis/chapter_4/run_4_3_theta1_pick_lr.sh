#!/bin/bash
# =============================================================================
# 4.3  θ¹_SimCLR — Step 2/3：挑選 θ¹ 的最佳 pretraining lr
#
#   把 3 個候選 checkpoint（lr∈{1e-4,2e-4,4e-4}, bs256/ep500, random init）
#   各自在 100% data 上 finetune（seed42、下游 lr 掃描、3 runs），
#   acc 最高者即為 θ¹ 的正式 pretraining lr。
#
#   結果寫到（與 θ² 分開，不互相覆蓋）：
#     classification/exp_results/classification_hard/cold_start_simclr_randinit/
#
#   用法（repo 根）：
#     DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_theta1_pick_lr.sh
#   看結果：
#     python3 thesis/chapter_4/aggregate_results.py --theta1
# =============================================================================
set -e
cd "$(dirname "$0")/../../classification"   # run_first_iter_simclr.py 相對路徑以 classification/ 為基準

DEVICE=${DEVICE:-"cuda:0"}
SIMCLR_LRS=${SIMCLR_LRS:-"5e-5 1e-4 2e-4 4e-4"}   # 候選 θ¹ pretraining lr（含你也跑的 5e-5）
RUNS=${RUNS:-3}
BS=${BS:-256}
EP=${EP:-500}
# 在低/高 portion 都比：預訓練品質差異在「低 portion」最明顯，100% 幾乎分不出來。
PORTIONS=${PORTIONS:-"10 100"}
# 各 portion 對應的下游 lr 掃描網格（沿用 weights_init 慣例）
down_lrs_for () { case "$1" in 10) echo "5e-5 1e-4 3e-4";; *) echo "1e-4 3e-4 5e-4 7e-4 1e-3";; esac; }

for slr in $SIMCLR_LRS; do
  slr=$(python3 -c "print(str(float('$slr')))")   # 2e-4→0.0002, 5e-5→5e-05（對齊存檔格式）
  ckpt="../SSL/simclr/ckpt/resnet18_random_simclr_lr${slr}_bs${BS}_ep${EP}.pkl"
  if [ ! -f "$ckpt" ]; then
    echo "!! 找不到 $ckpt — 請先跑 run_4_3_theta1_pretrain.sh"; continue
  fi
  for p in $PORTIONS; do
    for dlr in $(down_lrs_for "$p"); do
      for r in $(seq 1 "$RUNS"); do
        echo "θ¹ pick-lr: simclr_lr=$slr portion=$p down_lr=$dlr run=$r/$RUNS"
        python3 ./run_first_iter_simclr.py \
            --task_type hard --pretrained_weights simclr --simclr_init random \
            --simclr_lr "$slr" --simclr_bs "$BS" --simclr_ep "$EP" \
            --portion "$p" --seed 42 --lr "$dlr" --aug_factor 4 --device "$DEVICE" || true
      done
    done
  done
done

echo "完成。用 python3 thesis/chapter_4/aggregate_results.py --theta1 看 ρ=10 與 100 哪個 simclr_lr 最高。"
