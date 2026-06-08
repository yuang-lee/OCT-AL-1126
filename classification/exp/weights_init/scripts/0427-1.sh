SIMCLR_EPS="200" \
SIMCLR_BSS="256" \
SIMCLR_LR="0.0004" \
AUGS="aug4" \
MAX_RUN=3 \
RUNS=3 \
SEEDS="10 24 38 42 57" \
LRS="5e-5 1e-4 5e-4" \
PORTIONS="2.5" \
DEVICE="cuda:7" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh


SIMCLR_EPS="50" \
SIMCLR_BSS="16" \
SIMCLR_LR="0.0001" \
AUGS="aug4" \
MAX_RUN=6 \
RUNS=3 \
SEEDS="10 24 38 42 57" \
LRS="5e-5 1e-4 5e-4" \
PORTIONS="2.5" \
DEVICE="cuda:7" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh