# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **Active Learning on Skin OCT images** (master's thesis work). It has three coupled stages, each in its own top-level directory:

- `SSL/simclr/` — self-supervised SimCLR pretraining of a ResNet-18 backbone (also `SSL/auto_encoder/`).
- `classification/` — supervised classification + active-learning query strategies, initialized from random / ImageNet / SimCLR / autoencoder weights.
- `segmentation/` — 2D nuclei segmentation + active learning (separate, U-Net-style pipeline).

`thesis/` holds plotting/analysis code for the writeup. `ds/` holds datasets. Results are committed as JSON files under each stage's `exp_results/`, and plots are regenerated from those JSONs by scripts under `*/exp/`.

## Environment

```bash
conda activate oct-env        # or oct-AL-env per the READMEs
```
Python 3.10, torch 2.5.1 / torchvision 0.20.1 (CUDA 12.4). Plus `timm`, `einops`, `tensorboard`.

## Critical conventions

- **Always run scripts from the repository root.** Entry points do `sys.path.insert(0, os.getcwd())` and use absolute-from-root imports like `from classification.utils.data import get_data`. Running from inside a subdirectory breaks imports. (Some older README snippets show `cd classification` first — prefer running from root.)
- **`--device` is passed explicitly per run** as `cuda:N`. There is no auto-placement; parallel sweep scripts hard-code different GPUs. `gpu_map.md` records the physical→logical GPU index mapping for this machine — consult it, the visible `cuda:N` is not the physical card.
- **`.gitignore` excludes `**.pkl`, `ds/`, `dataset/`, `tb_logs/`.** Model checkpoints and raw data are NOT in git. Only result JSONs and figures are committed.
- **Classification `--task_type` selects the dataset and class count:** `easy`→2 class (`ds/classification/two_class`), `medium`→4 class (`four_class`), `hard`→7 class (`seven_class`). This mapping lives in `run_first_iter.py` / `run_AL.py`.

## Stage 1 — SimCLR pretraining

```bash
python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train \
    -a resnet18 --epochs 100 -b 128 --lr 0.0002 --device cuda:0
```
Checkpoints saved to `SSL/simclr/ckpt/resnet18_simclr_lr{lr}_bs{bs}_ep{ep}.pkl`; metrics logged as JSON in `SSL/simclr/json/` and to TensorBoard (`tensorboard --logdir SSL/simclr/tb_logs`). Note (from README): learning rate should be linearly scaled with batch size, per the SimCLR paper. A SimCLR `.pkl` is consumed downstream via `--pretrained_weights simclr --simclr_path <path>` and loaded into `ResNetSimCLR` with `strict=False`.

## Stage 2 — Classification

Two entry points, both from repo root:

```bash
# First/full training at a fixed label portion (the "cold start" baseline)
python3 classification/run_first_iter.py --task_type hard --portion 100 \
    --pretrained_weights simclr --simclr_path SSL/simclr/ckpt/<...>.pkl \
    --seed 42 --device cuda:0

# Active-learning loop: grow labeled set from portion_start→portion_end
python3 classification/run_AL.py --task_type hard --AL_strategy entropy \
    --pretrained_weights random --portion_start 5 --portion_end 60 \
    --portion_interval 2.5 --seed 42 --device cuda:0
```

- **AL strategies** live in `classification/AL_strategy/`: `uncertainty.py` (`conf`, `entropy`, `margin`), `diversity.py` (`coreset`), `hybrid.py` (`badge`), `mc.py` (MC-dropout). `run_AL.py` dispatches on `--AL_strategy`.
- `run_AL.py` uses a **single fixed learning rate across all portions** (it does not re-tune lr per AL iteration — see the banner comment at the top of the file).
- Results JSON path: `exp_results/classification_{task_type}/{cold_start|AL}_{pretrained_weights}/...`. The AL runner appends to an existing JSON if present, so reruns accumulate rather than overwrite.
- `classification/utils/data.py` splits the original `val/` into val+test and builds the offline `AugmentedDataset` (flip-based, controlled by `--aug_factor` 2/3/4 and `--flip_type`; optional online `--color_jitter`).
- Batch sweep drivers are in `classification/scripts/` (parallel background runs with Ctrl-C cleanup traps).

## Stage 3 — Segmentation

```bash
# Baseline at a data portion
python3 segmentation/run_first_iter.py --dataroot ./ds/segmentation \
    --phase train --portion 5 --seed 42 --device cuda:0

# Active learning
python3 segmentation/run_AL.py --AL_strategy nuclei_entropy \
    --dataroot ./ds/segmentation --portion_start 5 --portion_end 100 \
    --portion_interval 2.5 --seed 42 --device cuda:0

# Ensemble / BALD variant
python3 segmentation/run_AL_ensemble.py --dataroot ./ds/segmentation \
    --AL_strategy bald_ensemble_mean --n_models 5 --initial_seed 42 \
    --portion_start 5 --portion_end 100 --portion_interval 2.5 --device cuda:0
```
Grayscale input (`--input_nc 1`), binary output (`--output_nc 1`). Strategies in `segmentation/AL_strategy/` (`uncertainty.py`, `hybrid.py`, `bald_ensemble.py`); model/loss/training helpers in `segmentation/utils/`. Results in `segmentation/exp_results/`.

## Plotting / analysis

Figures are not hand-edited — they are regenerated from committed result JSONs by scripts under `classification/exp/{AL,data_aug,weights_init}/` and `thesis/plot/` (e.g. `python3 classification/exp/weights_init/plot_all.py`). When a result JSON changes, rerun the corresponding plot script rather than editing the PNG.

## Tests

There is no test suite. `test.py` and the various `test*.py` files under `ds/` are one-off data-prep / sanity scripts, not a CI test harness.
