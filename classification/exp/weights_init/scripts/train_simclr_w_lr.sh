#!/bin/bash
set -e

EPOCHS=(10)
# BATCHES=(16 32 128 256)
BATCHES=(128 256)

DEVICE="cuda:1"       # 預設 device
ANCHOR_BS=64          # 預設 anchor batch size
ANCHOR_LR=0.0002      # 預設 anchor learning rate

# 解析 named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --DEVICE)
            DEVICE="$2"
            shift
            ;;
        --ANCHOR_BS)
            ANCHOR_BS="$2"
            shift
            ;;
        --ANCHOR_LR)
            ANCHOR_LR="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: bash run_simclr.sh --DEVICE cuda:1 --ANCHOR_BS 64 --ANCHOR_LR 0.0002"
            exit 1
            ;;
    esac
    shift
done

echo "=================================================="
echo "SimCLR training script"
echo "Device        : $DEVICE"
echo "Anchor BS     : $ANCHOR_BS"
echo "Anchor LR     : $ANCHOR_LR"
echo "LR scaling    : linear, lr = anchor_lr * batch_size / anchor_bs"
echo "Epochs        : ${EPOCHS[*]}"
echo "Batch sizes   : ${BATCHES[*]}"
echo "=================================================="

for EPOCH in "${EPOCHS[@]}"; do
    for B in "${BATCHES[@]}"; do

        # linear scaling: lr = anchor_lr * batch_size / anchor_bs
        LR=$(python3 - <<EOF
anchor_lr = float("${ANCHOR_LR}")
batch_size = float("${B}")
anchor_bs = float("${ANCHOR_BS}")
lr = anchor_lr * batch_size / anchor_bs
print(f"{lr:.10g}")
EOF
)

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