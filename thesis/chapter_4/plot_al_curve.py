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
import os, sys, json, argparse, re
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
# 每類用同色調、不同細節色（如 Uncertainty 三層藍）：
#   Uncertainty=藍、Diversity=綠、Hybrid=紅、Baseline=灰。
# 顏色：同類同色調不同細節色（Uncertainty 藍 / Diversity 綠 / Hybrid 紅 / Baseline 灰）。
# marker：每個方法專屬、全圖互異（global 與單組圖一致）→ 顏色+形狀雙重編碼，最好分辨。
GROUPS = [
    ("Uncertainty", [
        ("conf",    "Confidence", "#08519C", "o"),   # ○
        ("margin",  "Margin",     "#3182BD", "s"),   # □
        ("entropy", "Entropy",    "#6BAED6", "^"),   # △
    ]),
    ("Diversity", [
        ("coreset",   "Coreset",   "#238B45", "D"),   # ◇ 深綠
        ("typiclust", "TypiClust", "#74C476", "v"),   # ▽ 淺綠
    ]),
    ("Hybrid", [
        ("badge",          "BADGE",          "#A50F15", "P"),   # ✚ 深紅
        ("cluster_margin", "Cluster-Margin", "#FB6A4A", "*"),   # ✦ 淺紅
    ]),
    ("Baseline",  [("random",  "Random",  "#404040", "X")]),   # ✕
]


def pool_strategy(strat, aug):
    if not os.path.isdir(AL_DIR):
        return {}
    pooled = {}
    for f in os.listdir(AL_DIR):
        if not f.endswith(".json") or "copy" in f or f.split("_seed")[0] != strat:
            continue   # 用 _seed 切，避免 'cluster_margin' 被 '_' 切錯
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


def _fmt_lr(lr):
    """'5e-05'→'5e-5'、'0.0001'→'1e-4'、'0.0003'→'3e-4'。"""
    try:
        return f"{float(lr):.0e}".replace("e-0", "e-").replace("e+0", "e+")
    except Exception:
        return str(lr)


def _per_seed_best(lrd):
    """單一 seed×portion 的 {lr: vals} → (best_acc%, best_lr)；無資料回 None。"""
    best = None
    for lr, v in lrd.items():
        a = _acc_list(v)
        if not a:
            continue
        m = float(np.mean(a)) * 100.0
        if best is None or m > best[0]:
            best = (m, lr)
    return best


def print_per_seed_tables(aug):
    """每個 strategy 印一張表：列=portion，欄=seed，格子= 'acc (best_lr)'。"""
    if not os.path.isdir(AL_DIR):
        print("AL_simclr/ 不存在 — 無 per-seed 結果可印。")
        return
    # strategy -> seed -> {portion: (acc, lr)}
    data = {}
    for f in os.listdir(AL_DIR):
        if not f.endswith(".json") or "copy" in f:
            continue
        strat = f.split("_seed")[0]   # 用 _seed 切，避免 'cluster_margin' 被 '_' 切錯
        m = re.search(r"seed(\d+)", f)
        if not m:
            continue
        seed = m.group(1)
        d = json.load(open(os.path.join(AL_DIR, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            b = _per_seed_best(lrd)
            if b:
                data.setdefault(strat, {}).setdefault(seed, {})[float(p)] = b
    if not data:
        print("AL_simclr/ 目前無任何 per-seed 結果。")
        return

    # strategy 順序：依 GROUPS，再補其他
    order = [k for _, items in GROUPS for k, *_ in items]
    strats = [s for s in order if s in data] + [s for s in data if s not in order]

    CW = 14  # 每個 seed 欄寬
    print("\n" + "=" * 78)
    print(f" AL per-seed results  (aug={aug})   格子 = test_acc(%) (best downstream lr)")
    print("=" * 78)
    for strat in strats:
        seeds = sorted(data[strat], key=lambda s: int(s))
        portions = sorted({p for sd in data[strat].values() for p in sd})
        print(f"\n■ {strat}")
        header = (f"{'ρ(%)':>6} | " + " ".join(f"seed{s:<{CW-4}}" for s in seeds)
                  + f" ‖ {'mean':>6} {'std':>5}")
        print(header)
        print("-" * len(header))
        for p in portions:
            cells = []
            accs = []
            for s in seeds:
                hit = data[strat][s].get(p)
                if hit:
                    cells.append(f"{hit[0]:4.1f} ({_fmt_lr(hit[1])})".ljust(CW))
                    accs.append(hit[0])
                else:
                    cells.append("—".center(CW))
            mean_s = float(np.mean(accs)) if accs else float("nan")
            std_s = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            agg = f" ‖ {mean_s:>6.1f} {std_s:>5.1f}" if accs else f" ‖ {'—':>6} {'—':>5}"
            print(f"{p:>6.1f} | " + " ".join(cells) + agg)
    print("=" * 78 + "\n")


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def _method_row(label, color, ls, marker, fontsize):
    da = DrawingArea(30, 14, 0, 0)
    da.add_artist(Line2D([3, 27], [7, 7], color=color, linewidth=3, linestyle=ls))
    if marker:
        da.add_artist(Line2D([15], [7], color=color, marker=marker, markersize=7, linestyle="None"))
    return HPacker(children=[da, TextArea(label, textprops=dict(fontsize=fontsize))],
                   align="center", pad=0, sep=5)


def grouped_legend(ax, columns, bottom_items=None, loc="lower right", fontsize=15):
    """多欄階層式 legend + 可選底部橫向 row。
    columns: [(groups, item_sep), ...]，groups=[(header,[(label,color,ls,marker),...]),...]；
             item_sep 控制群組內各方法的垂直行距。
    bottom_items: [(label,color,ls,marker), ...] 橫向排在所有欄下方（同一列、左對齊），無標題。"""
    def make_column(groups, item_sep):
        gboxes = []
        for header, items in groups:
            item_box = VPacker(children=[_method_row(*it, fontsize) for it in items],
                               align="left", pad=0, sep=item_sep)
            gbox = VPacker(children=[
                TextArea(header, textprops=dict(weight="bold", fontsize=fontsize)),
                DrawingArea(1, 3, 0, 0),
                item_box,
            ], align="left", pad=0, sep=2)
            gboxes.append(gbox)
        return VPacker(children=gboxes, align="left", pad=0, sep=8)

    col_boxes = [make_column(g, s) for (g, s) in columns if g]
    top = HPacker(children=col_boxes, align="top", pad=0, sep=16)

    children = [top]
    if bottom_items:
        children.append(DrawingArea(1, 10, 0, 0))   # 上方欄位與底部列之間留白
        bottom = HPacker(children=[_method_row(*it, fontsize) for it in bottom_items],
                         align="center", pad=0, sep=22)   # Random 與 Target 同列、相鄰
        children.append(bottom)
    box = VPacker(children=children, align="left", pad=2, sep=2)

    anchored = AnchoredOffsetbox(loc=loc, child=box, frameon=True, borderpad=0.5)
    anchored.patch.set_alpha(0.9)
    anchored.patch.set_edgecolor("0.7")
    ax.add_artist(anchored)


def draw_single_group(gname, items, out, aug):
    """單一類別一張圖：該類方法 + Random(灰虛) + Target(黑虛)，單欄簡單 legend（portion 風格）。
    顏色與 marker 都沿用 GROUPS（跨 global/單組圖一致）。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    drew = False
    for key, label, color, marker in items:
        data = pool_strategy(key, aug)
        if not data:
            print(f"    [skip] {key}: 無資料"); continue
        ps = sorted(data)
        mean = np.array([data[p][0] for p in ps]); std = np.array([data[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color,
                linewidth=3, markersize=9, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
        drew = True
    if not drew:
        plt.close(fig); print(f"  [skip group] {gname}: 無資料，不畫"); return
    # Random baseline（灰虛線）
    rb = random_baseline(aug)
    if rb:
        ps = sorted(rb); mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=9,
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.12)
    # Target（黑虛線）
    ax.axhline(y=88.2, color="black", linestyle=(0, (8, 4)), linewidth=2.2, alpha=0.85, label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")   # 單欄、portion_curve 風格
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs", "al_curve.png"))
    args = ap.parse_args()

    print_per_seed_tables(args.aug)

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

    # Target Performance 水平參考線（寫死 88.2%）
    TARGET = 88.2
    DASH = (0, (8, 4))
    ax.axhline(y=TARGET, color="black", linestyle=DASH, linewidth=2.2, alpha=0.85)
    item_target = ("Target", "black", DASH, None)
    # 併入 Baseline 群組（與 Random 同列）；若 Random 沒資料就自建 Baseline
    for h, baseitems in legend_groups:
        if h == "Baseline:":
            baseitems.append(item_target)
            break
    else:
        legend_groups.append(("Baseline:", [item_target]))
    print(f"  Target Performance line = {TARGET:.1f}%")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    # 兩欄等高；Random / Target 各掛在所屬欄底部（自動靠齊左緣，一左一右），不寫 "Baseline"
    by = dict(legend_groups)
    def _grp(h):
        return [(h, by[h])] if by.get(h) else []
    base = by.get("Baseline:", [])                        # [Random, Target]
    foot_random = next((it for it in base if it[0] == "Random"), None)
    foot_target = next((it for it in base if it[0] == "Target"), None)
    # 三欄：Uncertainty | Diversity | Hybrid（拆開）；Random 與 Target 同列橫排在最下方
    columns = [
        (_grp("Uncertainty:"), 4),
        (_grp("Diversity:"),   3),
        (_grp("Hybrid:"),      3),
    ]
    columns = [(g, s) for (g, s) in columns if g]
    bottom = [it for it in (foot_random, foot_target) if it]   # Random 左、Target 右
    grouped_legend(ax, columns, bottom_items=bottom, loc="lower right", fontsize=FONT_LEGEND)
    style_ax(ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.out}")

    # 另外三張：每類別單獨一張（單欄 legend、組內不同 marker、顏色/marker 跨圖一致）
    base, ext = os.path.splitext(args.out)
    for gname, items in GROUPS:
        if gname == "Baseline":
            continue
        print(f"\n— 單組圖：{gname} —")
        draw_single_group(gname, items, f"{base}_{gname.lower()}{ext}", args.aug)


if __name__ == "__main__":
    main()
