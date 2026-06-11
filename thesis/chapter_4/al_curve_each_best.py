#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.4 主動學習「各類最佳」曲線：從 Uncertainty / Diversity / Hybrid 三類中，
各挑出「最早達到 Target(88.2%) 的策略」，把這三條拉出來畫在同一張圖。
+ Random baseline（灰虛）+ Target（黑虛）。

「最早達到」= mean 曲線首次 ≥ 88.2% 的 ρ（在跨越的兩點間做線性內插，得到更細的交叉 ρ 來比較）；
若某類無任何策略達標，退而取「final/最高 mean 最高者」當代表（會在 terminal 標註未達標）。

legend 只寫「方法名稱」（不寫 Uncertainty / Diversity / Hybrid 類別字）。

資料/彙整邏輯完全沿用 plot_al_curve.py（per-seed best-lr → mean over seeds）。
用法（repo 根）：python3 thesis/chapter_4/al_curve_each_best.py
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_al_curve import (GROUPS, FONT_LABEL, pool_strategy, random_baseline,
                           style_ax)

TARGET = 88.2


def crossing_rho(curve, target=TARGET):
    """curve={portion:(mean,std)} → 首次達到 target 的 ρ（兩點間線性內插）；未達標回 (inf, max_mean)。
    回傳 (cross_rho, best_mean)：cross_rho 越小代表越早達標。"""
    ps = sorted(curve)
    means = [curve[p][0] for p in ps]
    best_mean = max(means) if means else float("-inf")
    for i, p in enumerate(ps):
        if means[i] >= target:
            if i == 0:
                return float(p), best_mean
            p0, m0 = ps[i - 1], means[i - 1]
            # 線性內插：m0→means[i] 跨越 target 的 ρ
            frac = (target - m0) / (means[i] - m0) if means[i] != m0 else 0.0
            return float(p0 + frac * (p - p0)), best_mean
    return float("inf"), best_mean


def pick_best_in_group(items, aug):
    """從一類的方法中，挑「最早達標」者；全未達標則挑 best_mean 最高者。
    回傳 (key, label, color, marker, curve, cross_rho, reached:bool) 或 None。"""
    cands = []
    for key, label, color, marker in items:
        curve = pool_strategy(key, aug)
        if not curve:
            print(f"    [skip] {key}: 無資料"); continue
        cross, best_mean = crossing_rho(curve)
        cands.append((key, label, color, marker, curve, cross, best_mean))
    if not cands:
        return None
    reached = [c for c in cands if np.isfinite(c[5])]
    if reached:                                   # 有達標者 → 取交叉 ρ 最小
        best = min(reached, key=lambda c: c[5])
        return (*best[:6], True)
    best = max(cands, key=lambda c: c[6])          # 全未達標 → 取最高 mean
    return (*best[:6], False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs", "al_curve_each_best.png"))
    args = ap.parse_args()

    # 三類各挑一條（排除 Baseline 類）
    chosen = []
    print("\n=== 各類最佳（最早達 88.2%）===")
    for gname, items in GROUPS:
        if gname == "Baseline":
            continue
        pick = pick_best_in_group(items, args.aug)
        if pick is None:
            print(f"  {gname:<12}: 無資料"); continue
        key, label, color, marker, curve, cross, reached = pick
        tag = f"ρ≈{cross:.1f}% 首次達標" if reached else f"未達標（max mean={max(v[0] for v in curve.values()):.2f}）"
        print(f"  {gname:<12}: {label:<14} ({tag})")
        chosen.append((label, color, marker, curve, reached))

    if not chosen:
        print("無任何策略資料 — 先跑 run_4_4_active_learning.sh"); return

    fig, ax = plt.subplots(figsize=(12, 8))
    for label, color, marker, curve, reached in chosen:
        ps = sorted(curve)
        mean = np.array([curve[p][0] for p in ps]); std = np.array([curve[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=9, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)

    # Random baseline（灰虛線）
    rb = random_baseline(args.aug)
    if rb:
        ps = sorted(rb); mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=9,
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.12)
    # Target（黑虛線）
    ax.axhline(y=TARGET, color="black", linestyle=(0, (8, 4)), linewidth=2.2, alpha=0.85, label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")   # 只寫方法名，無類別字
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
