#!/bin/bash
set -e

# 要掃的 SimCLR pretraining epochs
SIMCLR_EPS=(10 20 50 100 200 500)

# 要掃的 SimCLR pretraining batch sizes
SIMCLR_BSS=(16 32 64 128 256)

# 固定 downstream classifier 設定
AUGS_VAL="aug4"
MAX_RUN_VAL=3
RUNS_VAL=3
SEEDS_VAL="42"
LRS_VAL="5e-5"
PORTIONS_VAL="100"
DEVICE_VAL="cuda:0"
PRETRAINED_VAL="simclr"

SCRIPT="./exp/meta_scripts/train_parallel_simclr.sh"

for simclr_bs in "${SIMCLR_BSS[@]}"; do
    for simclr_ep in "${SIMCLR_EPS[@]}"; do
        echo ""
        echo "============================================================"
        echo "Running SimCLR grid:"
        echo "  SIMCLR_BS = ${simclr_bs}"
        echo "  SIMCLR_EP = ${simclr_ep}"
        echo "============================================================"

        AUGS="${AUGS_VAL}" \
        RUNS="${RUNS_VAL}" \
        MAX_RUN="${MAX_RUN_VAL}" \
        SEEDS="${SEEDS_VAL}" \
        LRS="${LRS_VAL}" \
        PORTIONS="${PORTIONS_VAL}" \
        DEVICE="${DEVICE_VAL}" \
        PRETRAINED="${PRETRAINED_VAL}" \
        SIMCLR_BS="${simclr_bs}" \
        SIMCLR_EP="${simclr_ep}" \
        "${SCRIPT}"

        echo "Finished SIMCLR_BS=${simclr_bs}, SIMCLR_EP=${simclr_ep}"
    done
done

echo ""
echo "============================================================"
echo "All SimCLR grid experiments completed."
echo "============================================================"