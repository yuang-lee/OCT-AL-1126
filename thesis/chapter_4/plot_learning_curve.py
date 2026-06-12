#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning 學習曲線：讀 thesis/chapter_4/learning_curves/{init}_p{portion}_s42.json，
每個 portion 一張圖（執行一次出 10% / 30% / 100% 三張），版面：
  (a) Training Loss   ┐ 左半上下，共用 x 軸（Fine-tuning Epoch）
  (b) Validation Loss ┘
  (c) Test Accuracy   → 右半，佔滿高度
四條折線 = 四種 weight init（θ_rand / θ_ImageNet / θ¹ / θ²，配色同 portion 曲線），無 marker。
legend 統一獨立畫在大圖最上方一排。風格對齊碩論（Arial / dpi300）。

用法（repo 根）：python3 thesis/chapter_4/plot_learning_curve.py
"""
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def moving_average(y, w):
    """置中移動平均（SMA, average smoothing）；邊緣以實際窗長正規化，不會把端點拉向 0。"""
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1 or y.size < 2:
        return y
    w = int(min(w, y.size))
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")

plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_TITLE = 22, 16, 19, 21

# (檔名前綴, legend 標籤, 顏色) — 配色與 plot_portion_curve.py 一致
INITS = [
    ("random",   r"$\theta_{\mathrm{rand}}$",       "#7F7F7F"),
    ("imagenet", r"$\theta_{\mathrm{ImageNet}}$",   "#2CA02C"),
    ("simclr1",  r"$\theta^{1}_{\mathrm{SimCLR}}$", "#E67E22"),
    ("simclr2",  r"$\theta^{2}_{\mathrm{SimCLR}}$", "#8E44AD"),
]


def load_seeds(curve_dir, prefix, portion):
    """讀同一 (init, portion) 的所有 seed JSON：{prefix}_p{portion}_s*.json → [dict, ...]（按 seed 排序）。"""
    files = sorted(glob.glob(os.path.join(curve_dir, f"{prefix}_p{portion}_s*.json")))
    return [json.load(open(f)) for f in files], files


def train_xy(d, stride=10, smooth=9):
    """每 step train loss → 每隔 stride 採樣（含 step0=訓練前）→ 移動平均平滑。
    x 換算成 epoch 單位（step/steps_per_epoch）。"""
    steps = d["train_steps"][::max(1, stride)]
    spe = sum(1 for s in d["train_steps"] if s["epoch"] == 1) or 1
    x = np.array([s["step"] / spe for s in steps])
    y = moving_average([s["train_loss"] for s in steps], smooth)
    return x, y


def epoch_xy(d, key, scale=1.0, smooth=3):
    eps = d["epochs"]
    x = np.array([e["epoch"] for e in eps])
    raw = np.array([e[key] * scale for e in eps], dtype=float)
    y = moving_average(raw, smooth)
    if y.size:                       # 保留 epoch0(訓練前起點) 與最後 epoch(收斂值) 真值，不被平滑糊掉
        y[0], y[-1] = raw[0], raw[-1]
    return x, y


def _stack(xys):
    """多個 seed 的 (x, y)（每 seed 先各自平滑）→ (x, mean_over_seeds, std_over_seeds)。
    seed 間以最短長度對齊；std 用 ddof=1（樣本標準差，n>=2 才有，單 seed 回傳 0）。"""
    n = min(len(x) for x, _ in xys)
    x = xys[0][0][:n]
    Y = np.vstack([y[:n] for _, y in xys])
    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros(n)
    return x, mean, std, Y.shape[0]


def train_curves(datas, stride, smooth):
    return _stack([train_xy(d, stride, smooth) for d in datas])


def epoch_curves(datas, key, scale, smooth):
    return _stack([epoch_xy(d, key, scale, smooth) for d in datas])


def plot_one(curve_dir, portion, out, stride=10, smooth_train=9, smooth_epoch=3):
    data = {p: load_seeds(curve_dir, p, portion)[0] for p, _, _ in INITS}
    present = [(p, lab, c) for (p, lab, c) in INITS if data[p]]
    if not present:
        print(f"  [skip] portion {portion}%：無 JSON"); return

    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15], height_ratios=[1, 1],
                           left=0.07, right=0.975, top=0.84, bottom=0.10, hspace=0.16, wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])                 # (a) train loss
    axB = fig.add_subplot(gs[1, 0], sharex=axA)     # (b) val loss（與 a 共用 x）
    axC = fig.add_subplot(gs[:, 1])                 # (c) test acc（右半全高）

    def band(ax, x, m, s, c, lw):
        ax.plot(x, m, color=c, linewidth=lw)
        if np.any(s > 0):                       # 多 seed 才畫 ±std 陰影（單 seed 無陰影）
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.18, linewidth=0)

    handles, n_ep, nseed_max = [], 0, 1
    for p, lab, c in present:
        datas = data[p]                          # 該 init 的所有 seed
        n_ep = max(n_ep, max(d["epochs"][-1]["epoch"] for d in datas))
        xa, ma, sa, ns = train_curves(datas, stride, smooth_train);          band(axA, xa, ma, sa, c, 1.8)
        xb, mb, sb, _  = epoch_curves(datas, "val_loss", 1.0, smooth_epoch);  band(axB, xb, mb, sb, c, 2.5)
        xc, mc, sc, _  = epoch_curves(datas, "test_acc", 100, smooth_epoch)
        axC.fill_between(xc, mc - sc, mc + sc, color=c, alpha=0.18, linewidth=0) if np.any(sc > 0) else None
        ln, = axC.plot(xc, mc, color=c, linewidth=2.5, label=lab)
        handles.append(ln)
        nseed_max = max(nseed_max, ns)

    axA.set_title("(a) Training Loss", fontsize=FONT_TITLE, pad=6)
    axB.set_title("(b) Validation Loss", fontsize=FONT_TITLE, pad=6)
    axC.set_title("(c) Test Accuracy", fontsize=FONT_TITLE, pad=6)
    axA.set_ylabel("Loss", fontsize=FONT_LABEL)
    axB.set_ylabel("Loss", fontsize=FONT_LABEL)
    axC.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axB.set_xlabel("Fine-tuning Epoch", fontsize=FONT_LABEL)   # a/b 共用：只在 b 標
    axC.set_xlabel("Fine-tuning Epoch", fontsize=FONT_LABEL)
    plt.setp(axA.get_xticklabels(), visible=False)             # a 隱藏 x ticklabels（與 b 共用）

    for ax in (axA, axB, axC):
        ax.set_xlim(0, n_ep)
        ax.set_xticks(range(0, n_ep + 1, 2))   # 0 2 4 6 8 ...（整數，不要 .0）
        ax.tick_params(labelsize=FONT_TICK, width=1.5, length=5)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
        for s in ax.spines.values():
            s.set_linewidth(1.5)

    # 三張圖（10/30/100%）共用同一 y 軸範圍與刻度，方便跨 portion 直接對比
    axA.set_ylim(0.0, 2.0); axA.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])      # (a) train loss
    axB.set_ylim(0.0, 3.0); axB.set_yticks([0, 1, 2, 3])                   # (b) val loss
    axC.set_ylim(0, 100);   axC.set_yticks([0, 20, 40, 60, 80, 100])      # (c) test acc

    # legend：統一獨立畫在大圖最上方一排
    fig.legend(handles=handles, loc="upper center", ncol=len(present),
               fontsize=FONT_LEGEND, frameon=True, edgecolor="0.7",
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.2, handlelength=2.4)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    nrep = {p: len(data[p]) for p, _, _ in present}
    word = "runs" if str(portion) == "100" else "seeds"   # ρ=100 是同 seed42 多 run，非多 seed
    print(f"  saved -> {out}  ({len(present)}/4 inits; {word} per init: "
          + ", ".join(f"{p}={n}" for p, n in nrep.items()) + ")")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--curve_dir", default=os.path.join(here, "learning_curves"))
    ap.add_argument("--out_dir", default=os.path.join(here, "figs"))
    ap.add_argument("--portions", nargs="+", default=["10", "30", "100"])
    ap.add_argument("--step_stride", type=int, default=10, help="train loss 每隔幾步採樣一點")
    ap.add_argument("--smooth_train", type=int, default=9, help="train loss 移動平均窗長（1=不平滑）")
    ap.add_argument("--smooth_epoch", type=int, default=3, help="val loss/test acc 移動平均窗長（1=不平滑）")
    args = ap.parse_args()
    for P in args.portions:
        plot_one(args.curve_dir, P, os.path.join(args.out_dir, f"learning_curve_p{P}.png"),
                 stride=args.step_stride, smooth_train=args.smooth_train, smooth_epoch=args.smooth_epoch)


if __name__ == "__main__":
    main()
