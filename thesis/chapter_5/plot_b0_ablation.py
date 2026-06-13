#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.1 b₀ ablation 畫圖：一個 AL 方法一張圖，把不同 b₀（初始隨機標註比例）畫在一起比較。
  - x 軸 = 標註比例 ρ（不同 b₀ 的軌跡從各自的 b₀ 起點開始；在「相同總 ρ」下對齊比較）。
  - 顏色 = 該方法在 4.4 的顏色（margin 藍 / coreset 綠 / badge 紅）；**b₀=2.5% 與 4.4 一模一樣**
    （同色同 marker），其餘 b₀ 同色、不同 marker、漸淺一點以利辨識。
  - 另畫 Random（passive，θ² cold-start）灰虛線 + Target 88.2% 黑虛線。

資料來源：
  - b₀=2.5%：4.4 主結果   classification/exp_results/classification_hard/AL_simclr/
  - b₀=5/10/20%：classification/exp_results/ch5_b0_ablation/b0_<B0>/classification_hard/AL_simclr/
彙整：per-seed best-lr → mean±std over seeds（與 4.4 同，沿用 plot_al_curve）。

用法（repo 根）：python3 thesis/chapter_5/plot_b0_ablation.py
"""
import os, sys, re, json, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from plot_al_curve import (_per_seed_best_curve, _per_seed_best, _fmt_lr, _acc_list,
                           random_baseline, style_ax, GROUPS, FONT_LABEL, FONT_TICK)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(REPO, "classification", "exp_results")

# 方法 → 4.4 顏色 / b₀=2.5 的 marker（保持與 4.4 一模一樣）
METHOD = {k: (c, m) for _, items in GROUPS for k, _l, c, m in items}
# 圖標題顯示名（與 4.4 legend 一致）
DISPLAY = {"margin": "Margin", "coreset": "Core-set", "badge": "BADGE",
           "cluster_margin": "Cluster-Margin", "conf": "Confidence", "entropy": "Entropy",
           "typiclust": "TypiClust"}

# 各方法給「一組手挑、彼此明顯區隔」的同系列配色（依 b₀/b 由小到大取 index）：
#   index 0 = 4.4 原色（最小 b₀/b 維持一模一樣）；之後每個都換色相+明度，相鄰一定分得出來。
#   margin 藍→靛→紫→洋紅、Core-set 綠→teal→青→藍、Cluster-Margin 珊瑚紅→紅→洋紅→紫。
PALETTE = {
    "margin":         ["#3182BD", "#5E50C0", "#9B2FB5", "#D6217E"],
    "coreset":        ["#238B45", "#2DD4BF", "#1D4ED8", "#7E22CE"],
    "cluster_margin": ["#FB6A4A", "#C81E1E", "#EC4899", "#7E2FB0"],   # 珊瑚→深紅→亮粉→紫
    "badge":          ["#A50F15", "#FB6A4A", "#EC4899", "#7E2FB0"],
}

def b0_dir(b0):
    if str(b0) == "2.5":
        return os.path.join(BASE, "classification_hard", "AL_simclr")            # 4.4 主結果
    return os.path.join(BASE, "ch5_b0_ablation", f"b0_{b0}", "classification_hard", "AL_simclr")

# (b₀, marker, linestyle, lighten 量)：三重編碼把不同 b₀ 拉開，但都還在「該方法同色系」內。
#   - 顏色：同一方法的 4.4 hue，往白漸淺以分層（b₀ 越大越淺）；b₀=2.5 = 4.4 原色（lighten=0）。
#   - marker：b₀=2.5 用該方法 4.4 marker；其餘刻意選「4.4 沒用到的形狀」(p 五邊形/h 六邊形/8 八邊形)，
#     不與 conf(o)/margin(s)/entropy(^)/coreset(D)/typiclust(v)/badge(P)/cluster_margin(*)/random(X) 撞。
#   - linestyle：再加不同虛實樣式。
# 只跑 b₀ = 2.5 / 10 / 20（不跑 5）。各 b₀ 一律實線，靠「色 + 形狀明顯不同的 marker」區分。
# marker 依 variant index：0=方法 4.4 marker、1=○、2=△、3=✚（圓/三角/十字 彼此差很大，不像多邊形那組相近）。
B0S = [("2.5", None, "-", 0.00),
       ("10", "o", "-", 0.55),
       ("20", "^", "-", 1.00)]

# 各 marker 形狀的視覺面積不同 → 用 per-shape 尺寸讓「看起來一樣大」。
MSIZE = {"s": 8, "D": 8.5, "P": 11, "*": 12, "o": 9.5, "^": 10, "v": 10, "X": 9,
         "p": 10.5, "h": 10.5, "8": 10.5}


def _hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def _lighten(hexc, amt):
    """往白色混 amt（0=原色，1=白）。"""
    return tuple(_hex2rgb(hexc) + (1.0 - _hex2rgb(hexc)) * amt)


def _blend(c1, c2, frac):
    """從 c1 往 c2 線性內插 frac（0=c1 原色，1=c2）。用於同系列的色相漸變。"""
    a, b = _hex2rgb(c1), _hex2rgb(c2)
    return tuple(a + (b - a) * frac)


def pool_strategy_dir(al_dir, strat, aug):
    """讀某個 AL_simclr 目錄下某策略的所有 seed → {portion: (mean%, std%)}（per-seed best-lr）。"""
    if not os.path.isdir(al_dir):
        return {}, {}
    by_portion, seeds_at = {}, {}
    for f in os.listdir(al_dir):
        if not f.endswith(".json") or "copy" in f or f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        seed = m.group(1) if m else f
        try:
            d = json.load(open(os.path.join(al_dir, f)))
        except Exception:
            continue
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            by_portion.setdefault(float(p), {})[seed] = lrd
            seeds_at.setdefault(float(p), set()).add(seed)
    return _per_seed_best_curve(by_portion), seeds_at


def per_seed_dir(al_dir, strat, aug):
    """{seed: {portion: (acc%, best_lr)}}：每個 seed 在各 portion 的 best-lr acc。"""
    out = {}
    if not os.path.isdir(al_dir):
        return out
    for f in os.listdir(al_dir):
        if not f.endswith(".json") or "copy" in f or f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        if not m:
            continue
        seed = m.group(1)
        try:
            d = json.load(open(os.path.join(al_dir, f)))
        except Exception:
            continue
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            b = _per_seed_best(lrd)
            if b:
                out.setdefault(seed, {})[float(p)] = b
    return out


def print_per_seed(method, b0, psd):
    """每個 (method, b₀) 一張表：列=ρ、欄=各 seed，格子 = acc (best-lr)，右側 mean/std。"""
    if not psd:
        print(f"\n  ■ {method} b₀={b0}%：無 per-seed 資料"); return
    seeds = sorted(psd, key=int)
    portions = sorted({p for sd in psd.values() for p in sd})
    CW = 15
    print(f"\n  ■ {method}  b₀={b0}%   格子 = test_acc(%) (best lr)")
    header = (f"  {'ρ(%)':>6} | " + " ".join(f"seed{s:<{CW-4}}" for s in seeds)
              + f" ‖ {'mean':>6} {'std':>5}")
    print(header); print("  " + "-" * (len(header) - 2))
    for p in portions:
        cells, accs = [], []
        for s in seeds:
            hit = psd[s].get(p)
            if hit:
                cells.append(f"{hit[0]:.2f} ({_fmt_lr(hit[1])})".ljust(CW))
                accs.append(hit[0])
            else:
                cells.append("—".center(CW))
        mu = float(np.mean(accs)) if accs else float("nan")
        sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        agg = f" ‖ {mu:>6.2f} {sd:>5.2f}" if accs else f" ‖ {'—':>6} {'—':>5}"
        print(f"  {p:>6.1f} | " + " ".join(cells) + agg)


def print_table(method, curves, seeds_at, rb):
    """結構化 terminal 輸出：列=ρ、欄=各 b₀（mean±std），最後一欄 Random。"""
    b0_keys = [b for b, *_ in B0S if curves.get(b)]
    all_p = sorted({p for b in b0_keys for p in curves[b]} | set(rb))
    CW = 12
    print("\n" + "=" * (8 + CW * (len(b0_keys) + 1)))
    print(f" b₀ ablation — {method}  (aug4)   cell = mean±std over seeds (per-seed best-lr)")
    print("=" * (8 + CW * (len(b0_keys) + 1)))
    head = f"{'ρ(%)':>6} |" + "".join(f"{'b0='+b:>{CW}}" for b in b0_keys) + f"{'Random':>{CW}}"
    print(head); print("-" * len(head))
    for p in all_p:
        cells = ""
        for b in b0_keys:
            if p in curves[b]:
                mu, sd = curves[b][p]
                cells += f"{mu:5.1f}±{sd:4.1f}".rjust(CW)
            else:
                cells += "—".rjust(CW)
        if p in rb:
            mu, sd = rb[p]; cells += f"{mu:5.1f}±{sd:4.1f}".rjust(CW)
        else:
            cells += "—".rjust(CW)
        print(f"{p:>6.1f} |" + cells)
    # 每個 b₀ 的 seed 覆蓋（提醒 n 不足的點）
    print("-" * len(head))
    for b in b0_keys:
        ns = {f"{p:g}": len(seeds_at[b].get(p, [])) for p in sorted(seeds_at[b])}
        print(f"  b0={b}% seeds/點: " + ", ".join(f"{k}:{v}" for k, v in ns.items()))


def plot_method(method, aug, out_dir):
    color, base_marker = METHOD[method]
    curves, seeds_at = {}, {}
    for b0, mk, ls, amt in B0S:
        c, s = pool_strategy_dir(b0_dir(b0), method, aug)
        if c:
            curves[b0] = c
            seeds_at[b0] = s
    rb = random_baseline(aug)
    if not curves:
        print(f"[skip] {method}：無任何 b₀ 資料"); return
    print_table(method, curves, seeds_at, rb)
    # 各 b₀ 的 per-seed 明細（哪個 seed、哪個 portion 有結果、各自的值）
    for b0, _mk, _ls, _amt in B0S:
        if b0 in curves:
            print_per_seed(method, b0, per_seed_dir(b0_dir(b0), method, aug))

    fig, ax = plt.subplots(figsize=(12, 8))
    pal = PALETTE.get(method, [color])
    for i, (b0, mk, ls, amt) in enumerate(B0S):
        if b0 not in curves:
            continue
        marker = base_marker if mk is None else mk         # b₀=2.5 用 4.4 marker
        col = pal[i] if i < len(pal) else pal[-1]          # index 0 = 4.4 原色；之後手挑色

        ps = sorted(curves[b0])
        mean = np.array([curves[b0][p][0] for p in ps])
        std = np.array([curves[b0][p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, linestyle=ls, color=col, linewidth=3,
                markersize=MSIZE.get(marker, 9), label=f"$b_0$={b0}%")
        ax.fill_between(ps, mean - std, mean + std, color=col, alpha=0.10)
    # Random baseline（灰虛）
    if rb:
        ps = sorted(rb); mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=MSIZE["X"],
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.10)
    # Target
    ax.axhline(y=88.2, color="black", linestyle=(0, (8, 4)), linewidth=2.2, alpha=0.85, label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.set_title(DISPLAY.get(method, method.capitalize()), fontsize=FONT_LABEL, pad=10)
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"b0_ablation_{method}.png")
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
