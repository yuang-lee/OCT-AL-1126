#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.3 portion 曲線：不同初始化 θ₀ 在各標註比例 ρ 下的分類準確率。
四條線（有資料才畫）：θ_rand / θ_ImageNet / θ¹_SimCLR(random→) / θ²_SimCLR(ImageNet→)。

- 與 aggregate_results.py 同一套讀法（best-lr per portion、跨 seed pool），數字一致。
- 兩種 SimCLR 預設都用最大設定 lr0.0002 / bs256 / ep500（可由參數改）。
- 論文 style、legend 用數學記號（對齊 Table 4-2）。

用法（repo 根）：
  python3 thesis/chapter_4/plot_portion_curve.py
  python3 thesis/chapter_4/plot_portion_curve.py --simclr_bs 256 --simclr_ep 500 --simclr_lr 0.0002
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_results import EXP, pool_seed_files   # 同邏輯，數字一致

FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 18
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})


# 論文官方 portion（與 Table 4-2 一致）；其餘 2.5-step 細點是舊探索資料，預設不畫
CANON = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--simclr_lr", default="0.0002")
    ap.add_argument("--simclr_bs", default="256")
    ap.add_argument("--simclr_ep", default="500")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs", "portion_curve.png"))
    ap.add_argument("--all_portions", action="store_true",
                    help="連 2.5-step 細點（舊探索資料）也畫；預設只畫 canonical portions")
    args = ap.parse_args()

    cfg = f"simclr_lr{args.simclr_lr}_simclr_bs{args.simclr_bs}_simclr_ep{args.simclr_ep}"

    # 每個 init -> {portion: (mean%, std%, lr, n)}（best-lr per portion, 跨 seed pool）
    series = {
        "rand":     (pool_seed_files(os.path.join(EXP, "cold_start_random"),
                     lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), args.aug),
                     r"$\theta_{\mathrm{rand}}$", "#7F7F7F", "s"),
        "imagenet": (pool_seed_files(os.path.join(EXP, "cold_start_imagenet"),
                     lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), args.aug),
                     r"$\theta_{\mathrm{ImageNet}}$", "#2CA02C", "^"),
        "theta1":   (pool_seed_files(os.path.join(EXP, "cold_start_simclr_randinit"),
                     lambda f: cfg in f, args.aug),
                     r"$\theta^{1}_{\mathrm{SimCLR}}$", "#E67E22", "o"),
        "theta2":   (pool_seed_files(os.path.join(EXP, "cold_start_simclr"),
                     lambda f: cfg in f, args.aug),
                     r"$\theta^{2}_{\mathrm{SimCLR}}$", "#8E44AD", "o"),
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    order = ["rand", "imagenet", "theta1", "theta2"]
    for key in order:
        data, label, color, marker = series[key]
        ps = sorted(p for p in data if args.all_portions or p in CANON)
        if not ps:
            print(f"  [skip] {key}: 無資料（{label}）")
            continue
        mean = np.array([data[p][0] for p in ps])
        std = np.array([data[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=10, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.15)
        print(f"  [ok]  {key}: {len(ps)} portions ({ps[0]:g}→{ps[-1]:g})")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
