#!/bin/bash

# 第一組實驗
SIMCLR_EPS="200 500" \
SIMCLR_BSS="16" \
SIMCLR_LR="0.0001" \
AUGS="aug4" \
MAX_RUN=6 \
RUNS=3 \
SEEDS="10 24 38 42 57" \
LRS="5e-5 1e-4 5e-4" \
PORTIONS="2.5" \
DEVICE="cuda:0" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh

# 第二組實驗
SIMCLR_EPS="200 500" \
SIMCLR_BSS="64" \
SIMCLR_LR="0.0002" \
AUGS="aug4" \
MAX_RUN=6 \
RUNS=3 \
SEEDS="10 24 38 42 57" \
LRS="5e-5 1e-4 5e-4" \
PORTIONS="2.5" \
DEVICE="cuda:0" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh
