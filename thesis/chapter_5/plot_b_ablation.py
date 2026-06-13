#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.1.2 b ablation 畫圖：一個 AL 方法一張圖，把不同 b（每輪查詢間隔）畫在一起比較（固定 b₀=2.5%）。
  - 顏色 = 該方法 4.4 色；b=2.5% 與 4.4 一模一樣（同色同 marker、實線）。
  - 其他 b 同色漸淺 + 4.4 沒用到的 marker(p 五邊形/h 六邊形)，一律實線、marker 同大小。
  - 另畫 Random（θ² cold-start）灰虛線 + Target 88.2% 黑虛線。
  - terminal 同樣印 structured output：① b 對照表(mean±std) ② 每個 b 的 per-seed 明細。

資料來源：
  - b=2.5%：4.4 主結果   classification/exp_results/classification_hard/AL_simclr/
  - b=5/10%：classification/exp_results/ch5_b_ablation/b_<B>/classification_hard/AL_simclr/
彙整：per-seed best-lr → mean±std over seeds（沿用 4.4）。

用法（repo 根）：python3 thesis/chapter_5/plot_b_ablation.py
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_b0_ablation import (pool_strategy_dir, per_seed_dir, PALETTE, MSIZE,
                              METHOD, DISPLAY, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from plot_al_curve import random_baseline, style_ax, FONT_LABEL, _fmt_lr

# (b, marker, idx)：b=2.5 = 4.4 原樣（marker=None→方法 4.4 marker、原色）。
# marker 依 index：0=方法 marker、1=○、2=△、3=✚（圓/三角/十字 彼此差很大，可同時容納三~四種 variant 不混淆）。
BVARIANTS = [("2.5", None, 0.00), ("5", "o", 0.40), ("10", "^", 0.70), ("20", "P", 1.00)]


def b_dir(b):
    if str(b) == "2.5":
        return os.path.join(BASE, "classification_hard", "AL_simclr")              # 4.4 主結果
    return os.path.join(BASE, "ch5_b_ablation", f"b_{b}", "classification_hard", "AL_simclr")


def print_summary(method, curves, seeds_at, rb):
    bs = [b for b, *_ in BVARIANTS if curves.get(b)]
    all_p = sorted({p for b in bs for p in curves[b]} | set(rb))
    CW = 12
    print("\n" + "=" * (8 + CW * (len(bs) + 1)))
    print(f" b ablation — {DISPLAY.get(method, method)}  (b₀=2.5%, aug4)   cell = mean±std over seeds")
    print("=" * (8 + CW * (len(bs) + 1)))
    head = f"{'ρ(%)':>6} |" + "".join(f"{'b='+b:>{CW}}" for b in bs) + f"{'Random':>{CW}}"
    print(head); print("-" * len(head))
    for p in all_p:
        cells = ""
        for b in bs:
            cells += (f"{curves[b][p][0]:5.1f}±{curves[b][p][1]:4.1f}".rjust(CW)
                      if p in curves[b] else "—".rjust(CW))
        cells += (f"{rb[p][0]:5.1f}±{rb[p][1]:4.1f}".rjust(CW) if p in rb else "—".rjust(CW))
        print(f"{p:>6.1f} |" + cells)
    print("-" * len(head))
    for b in bs:
        ns = {f"{p:g}": len(seeds_at[b].get(p, [])) for p in sorted(seeds_at[b])}
        print(f"  b={b}% seeds/點: " + ", ".join(f"{k}:{v}" for k, v in ns.items()))


def print_per_seed(method, b, psd):
    if not psd:
        print(f"\n  ■ {DISPLAY.get(method, method)} b={b}%：無 per-seed 資料"); return
    seeds = sorted(psd, key=int)
    portions = sorted({p for sd in psd.values() for p in sd})
    CW = 15
    print(f"\n  ■ {DISPLAY.get(method, method)}  b={b}%   格子 = test_acc(%) (best lr)")
    header = (f"  {'ρ(%)':>6} | " + " ".join(f"seed{s:<{CW-4}}" for s in seeds)
              + f" ‖ {'mean':>6} {'std':>5}")
    print(header); print("  " + "-" * (len(header) - 2))
    for p in portions:
        cells, accs = [], []
        for s in seeds:
            hit = psd[s].get(p)
            if hit:
                cells.append(f"{hit[0]:.2f} ({_fmt_lr(hit[1])})".ljust(CW)); accs.append(hit[0])
            else:
                cells.append("—".center(CW))
        mu = float(np.mean(accs)) if accs else float("nan")
        sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        agg = f" ‖ {mu:>6.2f} {sd:>5.2f}" if accs else f" ‖ {'—':>6} {'—':>5}"
        print(f"  {p:>6.1f} | " + " ".join(cells) + agg)


def plot_method(method, aug, out_dir):
    color, base_marker = METHOD[method]
    curves, seeds_at = {}, {}
    for b, mk, amt in BVARIANTS:
        c, s = pool_strategy_dir(b_dir(b), method, aug)
        if c:
            curves[b] = c; seeds_at[b] = s
    rb = random_baseline(aug)
    if not curves:
        print(f"[skip] {method}：無任何 b 資料"); return
    print_summary(method, curves, seeds_at, rb)
    for b, *_ in BVARIANTS:
        if b in curves:
            print_per_seed(method, b, per_seed_dir(b_dir(b), method, aug))

    fig, ax = plt.subplots(figsize=(12, 8))
    pal = PALETTE.get(method, [color])
    for i, (b, mk, amt) in enumerate(BVARIANTS):
        if b not in curves:
            continue
        marker = base_marker if mk is None else mk
        col = pal[i] if i < len(pal) else pal[-1]            # index 0 = 4.4 原色；之後手挑色
        ps = sorted(curves[b])
        mean = np.array([curves[b][p][0] for p in ps]); std = np.array([curves[b][p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, linestyle="-", color=col, linewidth=3,
                markersize=MSIZE.get(marker, 9), label=f"$b$={b}%")
        ax.fill_between(ps, mean - std, mean + std, color=col, alpha=0.10)
    if rb:
        ps = sorted(rb); mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=MSIZE["X"],
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.10)
    ax.axhline(y=88.2, color="black", linestyle=(0, (8, 4)), linewidth=2.2, alpha=0.85, label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.set_title(DISPLAY.get(method, method.capitalize()), fontsize=FONT_LABEL, pad=10)
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"b_ablation_{method}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--methods", nargs="+", default=["margin", "coreset", "cluster_margin"])
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "figs"))
    args = ap.parse_args()
    for m in args.methods:
        plot_method(m, args.aug, args.out_dir)


if __name__ == "__main__":
    main()
