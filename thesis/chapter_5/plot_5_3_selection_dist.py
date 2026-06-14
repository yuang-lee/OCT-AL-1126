#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sec. 5.3  主動學習選取影像之分析
============================================

在某個 portion 下，分析某個 AL 策略「累積選取的影像」在七個 label 上的分佈，
並與 random（= 整個 training set 的真實類別比例）做比較。

資料來源:
    classification/exp_results/classification_hard/AL_simclr/labeled_ids/
        <strategy>_seed<seed>_bs16.json
    每個檔案結構: { "<portion>": {"selected":[...], "cumulative":[...], ...}, ... }
    跨多個 seed 聚合成 mean ± std。

Baseline 設計:
    random sampling 的「期望」分佈 == 整個 training set 的類別比例。
    因此 baseline 直接用 dataset 的真實比例（analytic，無模擬雜訊），
    不另外畫成一組 bar，而是當參考線/基準。

兩種輸出:
    1. 分佈圖 (--plot dist) : AL 各類別的 share(%)，疊上 dataset baseline。
    2. 差異圖 (--plot diff) : AL 相對 baseline 的偏差，
                             --diff pp        百分點 (share_AL - share_base)         [預設]
                             --diff relative  相對% ((cnt_AL - cnt_base)/cnt_base*100)

從 repo root 執行:
    python3 thesis/chapter_5/plot_5_3_selection_dist.py \
        --strategy margin coreset typiclust --portion 25 --plot diff
圖存到 thesis/chapter_5/figs/。
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, DrawingArea, HPacker, VPacker, AnchoredOffsetbox
from collections import Counter
from torchvision import datasets

# --------------------------------------------------------------------------- #
# 路徑 / 常數
# --------------------------------------------------------------------------- #
DATA_DIR = "./ds/classification/seven_class"
LABELED_IDS_DIR = "./classification/exp_results/classification_hard/AL_simclr/labeled_ids"
OUT_DIR = "./thesis/chapter_5/figs"
DEFAULT_SEEDS = [10, 24, 38, 42, 57]
BATCH_TAG = "bs16"

# 字級 / 樣式：與 Chapter 4 AL 圖 (plot_al_curve.py) 完全一致
FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 15

# 類別名稱顯示用縮寫（七類較長的兩個縮短）
NAME_ABBR = {"Seborrhoeic keratosis": "SK", "Solar lentigo": "SL"}

# 策略 label + 顏色：與 Chapter 4 AL 折線圖 (plot_al_curve.py GROUPS) 完全一致。
STRATEGY_STYLE = {
    "conf":           ("Confidence",     "#08519C"),   # Uncertainty 深藍
    "margin":         ("Margin",         "#3182BD"),   # Uncertainty 中藍
    "entropy":        ("Entropy",        "#6BAED6"),   # Uncertainty 淺藍
    "coreset":        ("Core-set",       "#238B45"),   # Diversity 深綠
    "typiclust":      ("TypiClust",      "#74C476"),   # Diversity 淺綠
    "badge":          ("BADGE",          "#A50F15"),   # Hybrid 深紅
    "cluster_margin": ("Cluster-Margin", "#FB6A4A"),   # Hybrid 淺紅
}
BASELINE_COLOR = "#404040"   # 與 Ch4 Random 灰一致

# marker 對齊 Ch4 折線圖 (plot_al_curve.py GROUPS)
STRATEGY_MARKER = {
    "conf": "o", "margin": "s", "entropy": "^",
    "coreset": "D", "typiclust": "v",
    "badge": "P", "cluster_margin": "*",
}

# legend 分組：一族一欄 (column)，與 Ch4 折線圖 legend 一致
GROUP_DEF = [
    ("Uncertainty", ["conf", "margin", "entropy"]),
    ("Diversity",   ["coreset", "typiclust"]),
    ("Hybrid",      ["badge", "cluster_margin"]),
]


def _swatch_row(color, label, fontsize, marker=None, style="bar", linestyle="-"):
    """一個 legend 項：色塊(bar) 或 線條+marker(line) + 文字。"""
    da = DrawingArea(30, 14, 0, 0)
    if style == "line":
        da.add_artist(Line2D([2, 28], [7, 7], color=color, linewidth=3, linestyle=linestyle))
        if marker:
            da.add_artist(Line2D([15], [7], color=color, marker=marker, markersize=8, linestyle="None"))
    else:
        da.add_artist(Rectangle((3, 2), 24, 10, facecolor=color, edgecolor="none", alpha=0.85))
    return HPacker(children=[da, TextArea(label, textprops=dict(fontsize=fontsize))],
                   align="center", pad=0, sep=5)


def grouped_legend(ax, strategies, baseline_label=None, loc="upper right", fontsize=FONT_LEGEND,
                   style="bar", markers=None, header_suffix=""):
    """階層式 legend：Uncertainty/Diversity/Hybrid 各為一欄；baseline 另列底部。
    style='line' 用線條+marker（折線圖），='bar' 用色塊（柱狀圖）。"""
    markers = markers or {}
    cols = []
    for header, keys in GROUP_DEF:
        present = [k for k in keys if k in strategies]
        if not present:
            continue
        item_box = VPacker(
            children=[_swatch_row(STRATEGY_STYLE[k][1], STRATEGY_STYLE[k][0], fontsize,
                                  marker=markers.get(k), style=style) for k in present],
            align="left", pad=0, sep=3)
        cols.append(VPacker(children=[
            TextArea(header + header_suffix, textprops=dict(weight="bold", fontsize=fontsize)),
            DrawingArea(1, 3, 0, 0),
            item_box,
        ], align="left", pad=0, sep=2))
    if not cols:
        return
    children = [HPacker(children=cols, align="top", pad=0, sep=16)]
    if baseline_label:
        children.append(DrawingArea(1, 8, 0, 0))
        if style == "line":
            children.append(_swatch_row(BASELINE_COLOR, baseline_label, fontsize,
                                        style="line", linestyle=(0, (6, 3))))
        else:
            children.append(_swatch_row(BASELINE_COLOR, baseline_label, fontsize))
    box = VPacker(children=children, align="left", pad=3, sep=3)
    anchored = AnchoredOffsetbox(loc=loc, child=box, frameon=True, borderpad=0.5)
    anchored.patch.set_alpha(0.92)
    anchored.patch.set_edgecolor("0.7")
    ax.add_artist(anchored)


# --------------------------------------------------------------------------- #
# 資料載入
# --------------------------------------------------------------------------- #
def load_dataset_labels(data_dir=DATA_DIR):
    """回傳 (labels: list[int], class_names: list[str])，順序為 ImageFolder 字母序。"""
    ds = datasets.ImageFolder(os.path.join(data_dir, "train"))
    return list(ds.targets), list(ds.classes)


def counts_from_indices(indices, labels, num_classes):
    d = np.zeros(num_classes, dtype=int)
    for idx in indices:
        d[labels[idx]] += 1
    return d


def load_strategy_counts(strategy, portion, labels, num_classes,
                         seeds=DEFAULT_SEEDS, mode="cumulative",
                         labeled_dir=LABELED_IDS_DIR):
    """
    讀某策略在指定 portion、跨多 seed 的類別 count 矩陣。

    Returns:
        counts: np.ndarray, shape (n_found_seeds, num_classes)  每列為一個 seed 的 count
        found_seeds: list[int]
    """
    portion_key = str(float(portion))
    rows, found = [], []
    for s in seeds:
        path = os.path.join(labeled_dir, f"{strategy}_seed{s}_{BATCH_TAG}.json")
        if not os.path.exists(path):
            print(f"  [warn] missing file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if portion_key not in data:
            print(f"  [warn] portion {portion_key} not in {os.path.basename(path)} "
                  f"(have: {sorted(data.keys(), key=float)[:3]}...)")
            continue
        if mode not in data[portion_key]:
            print(f"  [warn] mode '{mode}' not in {os.path.basename(path)} @ {portion_key}")
            continue
        rows.append(counts_from_indices(data[portion_key][mode], labels, num_classes))
        found.append(s)
    if not rows:
        return np.empty((0, num_classes), dtype=int), []
    return np.array(rows), found


# --------------------------------------------------------------------------- #
# 統計 + terminal 輸出
# --------------------------------------------------------------------------- #
def to_shares(counts):
    """count 矩陣 -> 每列(seed)的 share(%) 矩陣。"""
    totals = counts.sum(axis=1, keepdims=True)
    return 100.0 * counts / np.where(totals == 0, 1, totals)


def baseline_shares(labels, num_classes):
    """dataset 真實類別比例(%) —— 即 random 的期望分佈。"""
    cnt = Counter(labels)
    base = np.array([cnt[i] for i in range(num_classes)], dtype=float)
    return 100.0 * base / base.sum()


def print_report(strategy, counts, found_seeds, class_names, base_share, mode):
    """terminal 印出該策略的詳細分佈表。"""
    num_classes = len(class_names)
    shares = to_shares(counts)                 # (n_seed, C)
    cnt_mean, cnt_std = counts.mean(0), counts.std(0, ddof=1) if len(counts) > 1 else (counts.mean(0), np.zeros(num_classes))
    sh_mean, sh_std = shares.mean(0), (shares.std(0, ddof=1) if len(shares) > 1 else np.zeros(num_classes))

    pp_diff = sh_mean - base_share                                   # 百分點
    # 相對% 用「期望 count」當分母，避免 share 與 count 混淆
    n_total = counts.sum(1).mean()
    base_cnt = base_share / 100.0 * n_total
    rel_diff = 100.0 * (cnt_mean - base_cnt) / np.where(base_cnt == 0, np.nan, base_cnt)

    label, _ = STRATEGY_STYLE.get(strategy, (strategy, None))
    print("\n" + "=" * 78)
    print(f"  Strategy : {strategy}  ({label})")
    print(f"  Mode     : {mode}   |  seeds = {found_seeds}  (n={len(found_seeds)})")
    print(f"  Selected : {n_total:.0f} / {len(class_names) and ''}images @ this portion")
    print("=" * 78)
    header = f"  {'class':22s} {'count(mean±std)':>16s} {'AL share%':>11s} {'base%':>7s} {'Δpp':>7s} {'rel%':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    # 依 dataset 比例由多到少排序，閱讀更直觀
    order = np.argsort(base_share)[::-1]
    for i in order:
        nm = NAME_ABBR.get(class_names[i], class_names[i])
        print(f"  {nm:22s} {cnt_mean[i]:6.1f}±{cnt_std[i]:4.1f}     "
              f"{sh_mean[i]:6.1f}±{sh_std[i]:4.1f} {base_share[i]:6.1f} "
              f"{pp_diff[i]:+6.1f} {rel_diff[i]:+7.1f}")
    print("  " + "-" * (len(header) - 2))
    over = [class_names[i] for i in order if pp_diff[i] > 1.0]
    under = [class_names[i] for i in order if pp_diff[i] < -1.0]
    print(f"  over-sampled  (Δpp > +1): {', '.join(NAME_ABBR.get(c, c) for c in over) or '—'}")
    print(f"  under-sampled (Δpp < -1): {', '.join(NAME_ABBR.get(c, c) for c in under) or '—'}")


# --------------------------------------------------------------------------- #
# 繪圖
# --------------------------------------------------------------------------- #
def _display_names(class_names):
    return [NAME_ABBR.get(c, c) for c in class_names]


def style_ax(ax):
    """tick / grid / spine 樣式，與 Ch4 style_ax 一致。"""
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def plot_distribution(results, class_names, base_share, portion, out_path):
    """各策略的 share(%) 分佈，疊上 dataset baseline。"""
    disp = _display_names(class_names)
    order = np.argsort(base_share)[::-1]
    x = np.arange(len(class_names))
    strategies = list(results.keys())
    n = len(strategies)
    width = 0.8 / max(n + 1, 1)   # +1 留給 baseline

    fig, ax = plt.subplots(figsize=(12, 8))   # 與論文其它 AL 圖一致
    # baseline 當第一組灰 bar
    ax.bar(x + (0 - (n + 1) / 2 + 0.5) * width, base_share[order], width,
           label="Random / Dataset", color=BASELINE_COLOR, alpha=0.85)
    for i, strat in enumerate(strategies):
        shares = to_shares(results[strat])
        mean, std = shares.mean(0)[order], (shares.std(0, ddof=1) if len(shares) > 1 else np.zeros(len(class_names)))[order]
        label, color = STRATEGY_STYLE.get(strat, (strat, "#666666"))
        ax.bar(x + ((i + 1) - (n + 1) / 2 + 0.5) * width, mean, width,
               yerr=std, capsize=4, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Share of selected samples (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([disp[i] for i in order], fontsize=FONT_TICK, rotation=45, ha="right")
    style_ax(ax)
    grouped_legend(ax, strategies, baseline_label="Random / Dataset")
    fig.tight_layout()
    _save(fig, out_path)


def plot_difference(results, class_names, base_share, portion, out_path,
                    diff="pp", labels_list=None, num_classes=None):
    """各策略相對 baseline 的偏差。diff='pp' 百分點 ; diff='relative' 相對%。"""
    disp = _display_names(class_names)
    order = np.argsort(base_share)[::-1]
    x = np.arange(len(class_names))
    strategies = list(results.keys())
    n = len(strategies)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(12, 8))   # 與論文其它 AL 圖一致
    for i, strat in enumerate(strategies):
        counts = results[strat]
        shares = to_shares(counts)
        if diff == "pp":
            vals = shares.mean(0) - base_share
            errs = shares.std(0, ddof=1) if len(shares) > 1 else np.zeros(len(class_names))
        else:  # relative %
            n_total = counts.sum(1).mean()
            base_cnt = base_share / 100.0 * n_total
            safe = np.where(base_cnt == 0, np.nan, base_cnt)
            vals = 100.0 * (counts.mean(0) - base_cnt) / safe
            errs = 100.0 * (counts.std(0, ddof=1) if len(counts) > 1 else np.zeros(len(class_names))) / safe
        label, color = STRATEGY_STYLE.get(strat, (strat, "#666666"))
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals[order], width, yerr=errs[order], capsize=4,
               label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1.5, alpha=0.8)
    ylab = ("Over / Under Sampling vs. Random (%)" if diff == "pp"
            else "Relative Difference from Random (%)")
    ax.set_ylabel(ylab, fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([disp[i] for i in order], fontsize=FONT_TICK, rotation=45, ha="right")
    style_ax(ax)
    grouped_legend(ax, strategies, loc="lower right")
    fig.tight_layout()
    _save(fig, out_path)


def available_portions(strategy, seeds=DEFAULT_SEEDS, labeled_dir=LABELED_IDS_DIR):
    """掃某策略所有 seed 檔，回傳出現過的 portion（float，遞增排序）。"""
    ps = set()
    for s in seeds:
        path = os.path.join(labeled_dir, f"{strategy}_seed{s}_{BATCH_TAG}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            ps.update(float(k) for k in json.load(f).keys())
    return sorted(ps)


def class_trend(strategy, class_idx, labels, num_classes, seeds, mode):
    """某策略下，target class 的 share(%) 隨 portion 變化。回傳 (xs, mean, std)。"""
    xs, means, stds = [], [], []
    for p in available_portions(strategy, seeds):
        counts, _ = load_strategy_counts(strategy, p, labels, num_classes, seeds=seeds, mode=mode)
        if len(counts) == 0:
            continue
        share = to_shares(counts)[:, class_idx]
        xs.append(p)
        means.append(share.mean())
        stds.append(share.std(ddof=1) if len(share) > 1 else 0.0)
    return np.array(xs), np.array(means), np.array(stds)


def plot_class_trend(strategies, class_idx, class_name, labels, num_classes,
                     base_val, seeds, mode, out_path):
    """橫軸 portion ρ、縱軸 target class 的 share(%)，多策略折線 + dataset baseline。"""
    disp = NAME_ABBR.get(class_name, class_name)
    fig, ax = plt.subplots(figsize=(12, 8))   # 與論文其它 AL 圖一致
    print("\n" + "=" * 78)
    print(f"  Trend of '{class_name}' share(%) vs portion   (mode={mode})")
    print(f"  dataset baseline = {base_val:.1f}%")
    print("=" * 78)
    for strat in strategies:
        xs, means, stds = class_trend(strat, class_idx, labels, num_classes, seeds, mode)
        if len(xs) == 0:
            print(f"  [skip] {strat}: no data")
            continue
        label, color = STRATEGY_STYLE.get(strat, (strat, "#666666"))
        marker = STRATEGY_MARKER.get(strat, "o")
        ax.plot(xs, means, marker=marker, color=color, linewidth=3, markersize=8, label=label)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.12)
        # terminal：印出第一個「明顯低於 baseline（< -1pp）」的 portion
        below = [f"{x:g}" for x, m in zip(xs, means) if m < base_val - 1.0]
        first_drop = below[0] if below else "—"
        print(f"  {label:16s}: start_below_baseline @ ρ={first_drop:>5s}%  "
              f"(min={means.min():.1f}% @ ρ={xs[int(np.argmin(means))]:g}%)")

    ax.axhline(base_val, color=BASELINE_COLOR, linestyle=(0, (8, 4)),
               linewidth=2.2, alpha=0.85, label="Random / Dataset")
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Share of Selected Samples (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(class_name, fontsize=FONT_LABEL, pad=14)
    ax.set_xticks(list(range(0, 61, 10)))
    ax.set_xlim(0, 61)
    style_ax(ax)
    ax.grid(True, axis="both", linestyle="--", alpha=0.4, linewidth=1.0)  # 折線圖兩軸都加 grid
    grouped_legend(ax, strategies, baseline_label="Random / Dataset", loc="upper right",
                   style="line", markers=STRATEGY_MARKER, header_suffix=":")
    fig.tight_layout()
    _save(fig, out_path)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"\n[saved] {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Sec 5.3 AL selection class-distribution analysis")
    ap.add_argument("--strategy", nargs="+", default=None,
                    help="一個或多個策略 (conf/margin/entropy/coreset/typiclust/badge/cluster_margin)；"
                         "不指定時 dist/diff 用全部七種、trend 用 margin/coreset/cluster_margin")
    ap.add_argument("--portion", type=float, default=22.5, help="labeled portion (%%)")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--mode", choices=["cumulative", "selected"], default="cumulative",
                    help="cumulative=此 portion 累積標記集 ; selected=此步新選")
    ap.add_argument("--plot", choices=["dist", "diff", "both", "trend"], default="both")
    ap.add_argument("--diff", choices=["pp", "relative"], default="pp")
    ap.add_argument("--class", dest="cls", default="Normal",
                    help="trend 模式要追蹤的類別（預設 Normal）")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    ALL_STRATEGIES = ["conf", "margin", "entropy", "coreset", "typiclust", "badge", "cluster_margin"]
    # 未指定 --strategy 時的預設：trend 用三條代表、dist/diff 用全部七種
    if args.strategy is None:
        args.strategy = (["margin", "coreset", "cluster_margin"] if args.plot == "trend"
                         else ALL_STRATEGIES)

    labels, class_names = load_dataset_labels()
    num_classes = len(class_names)
    base_share = baseline_shares(labels, num_classes)

    print("=" * 78)
    print(f"  Dataset: {len(labels)} train images, {num_classes} classes")
    print(f"  Random/dataset baseline share(%): "
          + ", ".join(f"{NAME_ABBR.get(c, c)}={base_share[i]:.1f}" for i, c in enumerate(class_names)))
    print("=" * 78)

    # ---- trend 模式：橫軸 portion、縱軸某類別 share，多策略折線 ----
    if args.plot == "trend":
        if args.cls not in class_names:
            print(f"[error] class '{args.cls}' 不存在；可選: {class_names}")
            return
        strategies = args.strategy
        class_idx = class_names.index(args.cls)
        tag = "all" if set(strategies) == set(STRATEGY_STYLE.keys()) else "_".join(strategies)
        out = os.path.join(args.out_dir, f"5_3_trend_{args.cls.replace(' ', '')}_{tag}_{args.mode}.png")
        plot_class_trend(strategies, class_idx, args.cls, labels, num_classes,
                         base_share[class_idx], args.seeds, args.mode, out)
        return

    results = {}
    for strat in args.strategy:
        counts, found = load_strategy_counts(strat, args.portion, labels, num_classes,
                                             seeds=args.seeds, mode=args.mode)
        if len(counts) == 0:
            print(f"[skip] no data for strategy '{strat}' @ {args.portion}%")
            continue
        results[strat] = counts
        print_report(strat, counts, found, class_names, base_share, args.mode)

    if not results:
        print("\nNo results to plot.")
        return

    p = f"{args.portion:g}"
    # 若畫了全部七種策略，檔名用 "all"；否則列出方法名
    tag = "all" if set(results.keys()) == set(STRATEGY_STYLE.keys()) else "_".join(results.keys())
    if args.plot in ("dist", "both"):
        plot_distribution(results, class_names, base_share, p,
                          os.path.join(args.out_dir, f"5_3_dist_{tag}_p{p}_{args.mode}.png"))
    if args.plot in ("diff", "both"):
        plot_difference(results, class_names, base_share, p,
                        os.path.join(args.out_dir, f"5_3_diff-{args.diff}_{tag}_p{p}_{args.mode}.png"),
                        diff=args.diff)


if __name__ == "__main__":
    main()
