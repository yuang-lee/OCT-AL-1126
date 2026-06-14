#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3-16  本研究之 OCT 皮膚影像分類資料集類別分布
=====================================================

直接從磁碟數每類影像數（避免手寫數字出錯），畫長條圖（依數量遞減排序）。
- 預設 = 全資料集（train + val 兩個資料夾；val 之後才在執行期切成 val+test）= 2541 張。
  （AL 的 training pool 只有 train split = 2032 張；用 --scope train 可改畫那個。）
- 顯示名稱：Normal→Healthy、Solar lentigo→SL、Seborrhoeic keratosis→SK。
- 樣式沿用 thesis/plot/class_dist.py 的深藍。

從 repo root 執行：
    python3 thesis/chapter_3/plot_dataset_distribution.py            # 全資料集 (2541)
    python3 thesis/chapter_3/plot_dataset_distribution.py --scope train   # training pool (2032)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

DATA_DIR = "./ds/classification/seven_class"
OUT_DIR = "./thesis/chapter_3/figs"
COLOR_MAIN = "#4C72B0"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DISPLAY = {"Normal": "Healthy", "Solar lentigo": "SL", "Seborrhoeic keratosis": "SK"}


def count_dir(d):
    return len([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]) if os.path.isdir(d) else 0


def class_counts(scope):
    """回傳 {class_name: count}。scope='all' = train+val；'train' = 只 train。"""
    train_root = os.path.join(DATA_DIR, "train")
    classes = sorted(os.listdir(train_root))
    counts = {}
    for c in classes:
        n = count_dir(os.path.join(train_root, c))
        if scope == "all":
            n += count_dir(os.path.join(DATA_DIR, "val", c))
        counts[c] = n
    return counts


def plot(scope, out_dir):
    counts = class_counts(scope)
    # 依數量遞減排序
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [DISPLAY.get(c, c) for c, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    print(f"scope={scope}  total={total}")
    for c, v in items:
        print(f"  {DISPLAY.get(c, c):10s} ({c:22s}): {v:5d}  {100*v/total:5.1f}%")

    plt.rcParams.update({
        "font.size": 16, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    })
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    rects = ax.bar(x, values, 0.6, color=COLOR_MAIN, edgecolor="black", linewidth=1.5, zorder=3)

    ax.set_ylabel("Number of Images", fontsize=20, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=15, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=14, pad=8)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, max(values) * 1.15)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for r in rects:
        h = r.get_height()
        ax.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color="black")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"oct_class_distribution_{scope}")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {base}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "train"], default="all",
                    help="all=全資料集(train+val, 2541) ; train=AL training pool(2032)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    plot(args.scope, args.out_dir)


if __name__ == "__main__":
    main()
