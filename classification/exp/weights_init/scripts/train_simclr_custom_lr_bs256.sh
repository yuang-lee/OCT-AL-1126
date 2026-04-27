#!/bin/bash
set -e

EPOCHS=(20 50 200)
BATCHES=(256)

DEVICE="cuda:3"       # 預設 device

# ============================================================
# 寫死每個 batch size 對應的 learning rate
# ============================================================

declare -A BS_TO_LR
BS_TO_LR[16]="0.0001"
BS_TO_LR[32]="0.00015" # 0.0001414
BS_TO_LR[64]="0.0002"
BS_TO_LR[128]="0.0003" # 0.0002828 
BS_TO_LR[256]="0.0004"

# ============================================================
# 解析 named arguments
# ============================================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --DEVICE)
            DEVICE="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: bash run_simclr.sh --DEVICE cuda:1"
            exit 1
            ;;
    esac
    shift
done

echo "=================================================="
echo "SimCLR training script"
echo "Device        : $DEVICE"
echo "Epochs        : ${EPOCHS[*]}"
echo "Batch sizes   : ${BATCHES[*]}"
echo "LR mode       : manually specified by batch size"
echo "=================================================="

for B in "${BATCHES[@]}"; do
    echo "BS=$B, LR=${BS_TO_LR[$B]}"
done

echo "=================================================="

for EPOCH in "${EPOCHS[@]}"; do
    for B in "${BATCHES[@]}"; do

        LR="${BS_TO_LR[$B]}"

        if [[ -z "$LR" ]]; then
            echo "Error: No learning rate specified for batch size $B"
            exit 1
        fi

        echo ""
        echo "--------------------------------------------------"
        echo "Now training SimCLR"
        echo "Epochs      : $EPOCH"
        echo "Batch size  : $B"
        echo "LR          : $LR"
        echo "Device      : $DEVICE"
        echo "Command     : python3 ./SSL/simclr/run.py --epochs $EPOCH -b $B --device $DEVICE --lr $LR"
        echo "--------------------------------------------------"

        python3 ./SSL/simclr/run.py \
            --epochs "$EPOCH" \
            -b "$B" \
            --device "$DEVICE" \
            --lr "$LR"
    done
done