"""
4.5 綜合比較（factor ablation）彙整 —— 印出 Table 1 / Table 2（rows ρ=10/30/50）。

7 個 cell（Aug × Init × AL；AL 代表策略 = margin（4.4 最佳，user 定案 2026-06-12），
--strategy 可換）：
  aug_only   (aug4,   ImageNet, passive) ← 4.2 cold_start_imagenet（主實驗）
  init_only  (no_aug, θ²,       passive) ← chapter4_5_ablation/init_only
  al_only    (no_aug, ImageNet, AL)      ← chapter4_5_ablation/al_only
  all_three  (aug4,   θ²,       AL)      ← 4.4 AL_simclr/{strategy}（主實驗）
  wo_aug     (no_aug, θ²,       AL)      ← chapter4_5_ablation/wo_aug
  wo_init    (aug4,   ImageNet, AL)      ← chapter4_5_ablation/wo_init
  wo_al      (aug4,   θ²,       passive) ← 4.3 cold_start_simclr best cfg（主實驗）
另印參考列 none (no_aug, ImageNet, passive) ← 4.2 的 no_aug key（三策略全關的 baseline）。

慣例 = 全論文統一（2026-06-10）：per-seed best-lr（該 seed 自己 runs 的平均挑 lr）
→ mean±std(ddof=1) over seeds。缺資料的 cell 標 NA、seed 不足 5 會標 (n=...)。

用法（repo 根）：python3 thesis/chapter_4/aggregate_4_5.py [--strategy xxx]
"""
import argparse
import json
import os

import numpy as np

MAIN = "./classification/exp_results/classification_hard"
ABL = "./classification/exp_results/chapter4_5_ablation"
SEEDS = [10, 24, 38, 42, 57]
PORTIONS = [10.0, 30.0, 50.0]
THETA2_CFG = "simclr_lr0.0002_simclr_bs256_simclr_ep500"


def cell_files(strategy):
    """每個 cell：(seed → json path, aug_key)。"""
    cs_theta2 = f"random{{s}}_bs16_ep20_{THETA2_CFG}.json"
    al = f"{strategy}_seed{{s}}_bs16.json"
    return {
        "none":      (f"{MAIN}/cold_start_imagenet/random{{s}}_bs16_ep20.json", "no_aug"),
        "aug_only":  (f"{MAIN}/cold_start_imagenet/random{{s}}_bs16_ep20.json", "aug4"),
        "init_only": (f"{ABL}/init_only/classification_hard/cold_start_simclr/{cs_theta2}", "no_aug"),
        "al_only":   (f"{ABL}/al_only/classification_hard/AL_imagenet/{al}", "no_aug"),
        "all_three": (f"{MAIN}/AL_simclr/{al}", "aug4"),
        "wo_aug":    (f"{ABL}/wo_aug/classification_hard/AL_simclr/{al}", "no_aug"),
        "wo_init":   (f"{ABL}/wo_init/classification_hard/AL_imagenet/{al}", "aug4"),
        "wo_al":     (f"{MAIN}/cold_start_simclr/{cs_theta2}", "aug4"),
    }


def per_seed_best(path_tpl, aug_key, portion):
    """per-seed best-lr → list of representative accs（一 seed 一值）。"""
    vals = []
    for s in SEEDS:
        f = path_tpl.format(s=s)
        if not os.path.isfile(f):
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        lr_dict = d.get(aug_key, {}).get(str(portion), {})
        best = None
        for v in lr_dict.values():
            acc = v["acc"] if isinstance(v, dict) else v   # AL JSON 是 {"acc":[...]}，cold-start 是 list
            if len(acc):
                m = float(np.mean(acc))
                best = m if best is None or m > best else best
        if best is not None:
            vals.append(best)
    return vals


def fmt(vals):
    if not vals:
        return "NA"
    m = np.mean(vals) * 100
    if len(vals) == 1:
        return f"{m:.2f} (n=1)"
    sd = np.std(vals, ddof=1) * 100
    out = f"{m:.2f}±{sd:.2f}"
    if len(vals) < len(SEEDS):
        out += f" (n={len(vals)})"
    return out


def print_table(title, columns, cells):
    print(f"\n=== {title} ===")
    width = 22
    header = f"{'ρ (%)':<8}" + "".join(f"{c:<{width}}" for c, _ in columns)
    print(header)
    print("-" * len(header))
    for p in PORTIONS:
        row = f"{p:<8g}"
        for _, key in columns:
            path_tpl, aug_key = cells[key]
            row += f"{fmt(per_seed_best(path_tpl, aug_key, p)):<{width}}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="margin",
                    help="AL 代表策略（margin = 4.4 最佳，user 定案 2026-06-12）")
    args = ap.parse_args()
    cells = cell_files(args.strategy)

    print(f"4.5 factor ablation  (AL strategy = {args.strategy};  "
          f"per-seed best-lr → mean±std ddof=1 over seeds, ×100)")

    print_table("Table 1 — 一次只開一個策略",
                [("Data Aug (4x)", "aug_only"),
                 ("Weight Init (θ²)", "init_only"),
                 (f"AL ({args.strategy})", "al_only"),
                 ("All Three", "all_three")], cells)

    print_table("Table 2 — 一次只關一個策略",
                [("w/o Data Aug", "wo_aug"),
                 ("w/o Weight Init", "wo_init"),
                 ("w/o AL", "wo_al"),
                 ("All Three", "all_three")], cells)

    print_table("參考 — 三策略全關 baseline",
                [("None (no_aug+ImageNet)", "none")], cells)

    print("\n資料來源：aug_only=4.2、wo_al=4.3 θ²、all_three=4.4（主實驗）；"
          "其餘 4 arm 在 classification/exp_results/chapter4_5_ablation/。")
    print("補跑：ARM=init_only|al_only|wo_aug|wo_init DEVICE=cuda:N "
          "./thesis/chapter_4/run_4_5_ablation.sh")


if __name__ == "__main__":
    main()
