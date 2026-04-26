#!/bin/bash
set -e

# ============================================================
# 可由外部環境變數覆蓋的設定
# ============================================================

# 要掃的 SimCLR pretraining epochs
SIMCLR_EPS=(${SIMCLR_EPS:-500 100 200 10 20 50})

# 要掃的 SimCLR pretraining batch sizes
SIMCLR_BSS=(${SIMCLR_BSS:-256 128 16 32 64})

# Downstream classifier settings
AUGS=${AUGS:-"aug4"}
MAX_RUN=${MAX_RUN:-3}
RUNS=${RUNS:-3}
SEEDS=${SEEDS:-"42"}
LRS=${LRS:-"5e-5"}
PORTIONS=${PORTIONS:-"10"}
DEVICE=${DEVICE:-"cuda:5"}
PRETRAINED=${PRETRAINED:-"simclr"}

SCRIPT=${SCRIPT:-"./exp/meta_scripts/train_parallel_simclr.sh"}

# ============================================================
# 顯示目前設定
# ============================================================

echo ""
echo "============================================================"
echo "Wrapper configuration"
echo "============================================================"
echo "SIMCLR_EPS : ${SIMCLR_EPS[*]}"
echo "SIMCLR_BSS : ${SIMCLR_BSS[*]}"
echo "AUGS       : ${AUGS}"
echo "MAX_RUN    : ${MAX_RUN}"
echo "RUNS       : ${RUNS}"
echo "SEEDS      : ${SEEDS}"
echo "LRS        : ${LRS}"
echo "PORTIONS   : ${PORTIONS}"
echo "DEVICE     : ${DEVICE}"
echo "PRETRAINED : ${PRETRAINED}"
echo "SCRIPT     : ${SCRIPT}"
echo "============================================================"

# ============================================================
# 執行 grid search
# ============================================================

for simclr_bs in "${SIMCLR_BSS[@]}"; do
    for simclr_ep in "${SIMCLR_EPS[@]}"; do
        echo ""
        echo "============================================================"
        echo "Running SimCLR grid:"
        echo "  SIMCLR_BS = ${simclr_bs}"
        echo "  SIMCLR_EP = ${simclr_ep}"
        echo "============================================================"

        AUGS="${AUGS}" \
        RUNS="${RUNS}" \
        MAX_RUN="${MAX_RUN}" \
        SEEDS="${SEEDS}" \
        LRS="${LRS}" \
        PORTIONS="${PORTIONS}" \
        DEVICE="${DEVICE}" \
        PRETRAINED="${PRETRAINED}" \
        SIMCLR_BS="${simclr_bs}" \
        SIMCLR_EP="${simclr_ep}" \
        bash "${SCRIPT}"

        echo "Finished SIMCLR_BS=${simclr_bs}, SIMCLR_EP=${simclr_ep}"
    done
done

echo ""
echo "============================================================"
echo "All SimCLR grid experiments completed."
echo "============================================================"