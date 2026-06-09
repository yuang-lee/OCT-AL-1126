#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.4 主動學習曲線：Random baseline + 五種 AL 策略 × ρ（2.5% interval）。
配色分組：Uncertainty(藍色系) / Diversity / Hybrid / Baseline；
階層式 legend（類別標題粗體、貼齊 box 最左邊，用 offsetbox 自排）；
另加水平線 = θ² 在 ρ=100% full fine-tune 的結果。

- 與 aggregate_results.py 同邏輯（best-lr per ρ、跨 seed pool）。
- AL 策略讀 AL_simclr/{strategy}_seed{seed}_bs16.json；
  Random = θ² cold-start（隨機選子集）曲線。

用法（repo 根）：python3 thesis/chapter_4/plot_al_curve.py
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, VPacker, DrawingArea, AnchoredOffsetbox

sys.path.insert(0, os.path.dirname(__file__))
from aggregate_results import EXP, best_lr_per_portion, _acc_list, pool_seed_files

FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 15
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
AL_DIR = os.path.join(EXP, "AL_simclr")

# 分組：(類別, [(key, 顯示名, 顏色, marker)])
GROUPS = [
    ("Uncertainty", [
        ("conf",    "Confidence", "#08519C", "o"),
        ("margin",  "Margin",     "#3182BD", "o"),
        ("entropy", "Entropy",    "#6BAED6", "o"),
    ]),
    ("Diversity", [("coreset", "Coreset", "#2CA02C", "s")]),
    ("Hybrid",    [("badge",   "BADGE",   "#D62728", "D")]),
    ("Baseline",  [("random",  "Random",  "#404040", "X")]),
]


def pool_strategy(strat, aug):
    if not os.path.isdir(AL_DIR):
        return {}
    pooled = {}
    for f in os.listdir(AL_DIR):
        if not f.endswith(".json") or "copy" in f or f.split("_")[0] != strat:
            continue
        d = json.load(open(os.path.join(AL_DIR, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            for lr, v in lrd.items():
                pooled.setdefault(float(p), {}).setdefault(lr, []).extend(_acc_list(v))
    return best_lr_per_portion(pooled)


# Random baseline 只畫有跑完整的 portion（含 2.5；15/25/35/45/55 未跑完整，跳過）
RANDOM_PORTIONS = {2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0}


def random_baseline(aug):
    """Random(passive) baseline = θ² cold-start（隨機選子集）。
    2.5/10/100 等多 seed 點用 per-cfg pool；其餘（5,20,...,60）用 seed42 彙整檔。"""
    cs = os.path.join(EXP, "cold_start_simclr")
    out = {}
    # 來源1：seed42 彙整檔（有 5,15,20,...,70，但無 2.5）
    f = os.path.join(cs, "random42_bs16.json")
    if os.path.isfile(f):
        d = json.load(open(f))
        if aug in d:
            out.update({p: v for p, v in best_lr_per_portion(d[aug]).items() if p in RANDOM_PORTIONS})
    # 來源2：best-cfg 多 seed（含 2.5）→ 優先覆蓋
    st2 = pool_seed_files(cs, lambda fn: "simclr_lr0.0002_simclr_bs256_simclr_ep500" in fn, aug)
    out.update({p: v for p, v in st2.items() if p in RANDOM_PORTIONS})
    return out


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def grouped_legend(ax, groups, loc="lower right", fontsize=15):
    """自排階層式 legend：類別標題粗體、貼齊最左；策略 = 線+marker 後接文字。
    groups: [(header, [(label, color, linestyle, marker), ...]), ...]"""
    rows = []
    for header, items in groups:
        rows.append(TextArea(header, textprops=dict(weight="bold", fontsize=fontsize)))
        for label, color, ls, marker in items:
            da = DrawingArea(30, 14, 0, 0)
            da.add_artist(Line2D([3, 27], [7, 7], color=color, linewidth=3, linestyle=ls))
            if marker:
                da.add_artist(Line2D([15], [7], color=color, marker=marker,
                                     markersize=7, linestyle="None"))
            row = HPacker(children=[da, TextArea(label, textprops=dict(fontsize=fontsize))],
                          align="center", pad=0, sep=5)
            rows.append(row)
    box = VPacker(children=rows, align="left", pad=2, sep=3)
    anchored = AnchoredOffsetbox(loc=loc, child=box, frameon=True, borderpad=0.4)
    anchored.patch.set_alpha(0.9)
    anchored.patch.set_edgecolor("0.7")
    ax.add_artist(anchored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs", "al_curve.png"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12, 8))
    legend_groups = []
    drew = False

    for gname, items in GROUPS:
        drawn = []
        for key, label, color, marker in items:
            data = random_baseline(args.aug) if key == "random" else pool_strategy(key, args.aug)
            if not data:
                print(f"  [skip] {key}: 無資料")
                continue
            ps = sorted(data)
            mean = np.array([data[p][0] for p in ps])
            std = np.array([data[p][1] for p in ps])
            ls = "--" if key == "random" else "-"
            ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=8, linestyle=ls)
            ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
            drawn.append((label, color, ls, marker))
            print(f"  [ok]  {key}: {len(ps)} 點 ({ps[0]:g}→{ps[-1]:g})")
            drew = True
        if drawn:
            legend_groups.append((f"{gname}:", drawn))

    if not drew:
        print("AL_simclr/ 目前無任何策略資料 — 先跑 run_4_4_active_learning.sh"); return

    # θ² 在 ρ=100% full fine-tune 的水平參考線
    t2 = pool_seed_files(os.path.join(EXP, "cold_start_simclr"),
                         lambda f: "simclr_lr0.0002_simclr_bs256_simclr_ep500" in f, args.aug)
    if 100.0 in t2:
        y100 = t2[100.0][0]
        DASH = (0, (8, 4))
        ax.axhline(y=y100, color="black", linestyle=DASH, linewidth=2.2, alpha=0.85)
        item_100 = ("100% full data", "black", DASH, None)
        # 併入 Baseline 群組（與 Random 同列）；若 Random 沒資料就自建 Baseline
        for h, baseitems in legend_groups:
            if h == "Baseline:":
                baseitems.append(item_100)
                break
        else:
            legend_groups.append(("Baseline:", [item_100]))
        print(f"  θ² @100% full FT = {y100:.2f}")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    grouped_legend(ax, legend_groups, loc="lower right", fontsize=FONT_LEGEND)
    style_ax(ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
