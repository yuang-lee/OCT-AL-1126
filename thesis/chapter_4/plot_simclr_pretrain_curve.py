#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
畫 SimCLR 預訓練曲線（雙子圖，論文 style）：
  (a) 左：InfoNCE loss 隨 epoch 下降
  (b) 右：contrastive Top-1 / Top-5 accuracy 隨 epoch 上升

兩種模式：
  單一：  --init theta2 --bs 256 --ep 500
  對比：  --compare --bs 256 --ep 500           # θ¹(random init) vs θ²(ImageNet init) overlay

資料來源：SSL/simclr/json/{arch}_simclr_lr{lr}_bs{bs}_ep{ep}.json
  history = [{epoch, loss, top1, top5, lr}, ...]（每 epoch 一筆，pretrain 中亦可讀到目前進度）

用法（repo 根）：
  python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --init theta2 --bs 256 --ep 500
  python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --compare --bs 256 --ep 500
"""
import os, json, argparse
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---- 論文 style（對齊 thesis/plot 與 exp/weights_init/plot_simclr.py）----
FONT_LABEL, FONT_TICK, FONT_TITLE, FONT_LEGEND = 26, 20, 24, 16
plt.rcParams.update({
    "font.size": 16,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.5,
})
C_LOSS, C_TOP1, C_TOP5 = "#C0392B", "#2980B9", "#27AE60"
C_T1, C_T2 = "#E67E22", "#2980B9"   # θ¹ 橘 / θ² 藍（compare 模式）


def arch_prefix(init):
    if init in ("theta2", "imagenet"):
        return "resnet18_simclr", r"$\theta^{2}_{SimCLR}$ (ImageNet init)"
    if init in ("theta1", "random"):
        return "resnet18_random_simclr", r"$\theta^{1}_{SimCLR}$ (random init)"
    raise ValueError(f"unknown init: {init}")


def load_history(init, lr, bs, ep, json_dir):
    prefix, label = arch_prefix(init)
    fpath = os.path.join(json_dir, f"{prefix}_lr{lr}_bs{bs}_ep{ep}.json")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"找不到 {fpath}\n（init={init} → 前綴 {prefix}；確認該設定已 pretrain 過）")
    hist = json.load(open(fpath))["history"]
    return {
        "label": label,
        "ep":   [h["epoch"] for h in hist],
        "loss": [h["loss"]  for h in hist],
        "top1": [h["top1"]  for h in hist],
        "top5": [h["top5"]  for h in hist],
    }


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def plot_single(d, bs, ep, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    axL.plot(d["ep"], d["loss"], color=C_LOSS, linewidth=3)
    axL.set_ylabel("InfoNCE Loss", fontsize=FONT_LABEL, labelpad=8)
    axL.set_title("(a) Contrastive Loss", fontsize=FONT_TITLE, pad=10)

    axR.plot(d["ep"], d["top1"], color=C_TOP1, linewidth=3, label="Top-1")
    axR.plot(d["ep"], d["top5"], color=C_TOP5, linewidth=3, label="Top-5")
    axR.set_ylabel("Contrastive Accuracy (%)", fontsize=FONT_LABEL, labelpad=8)
    axR.set_title("(b) Instance-Discrimination Accuracy", fontsize=FONT_TITLE, pad=10)
    axR.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="lower right")

    for ax in (axL, axR):
        ax.set_xlabel("Pretraining Epoch", fontsize=FONT_LABEL, labelpad=8)
        style_ax(ax)
    _save(fig, out)


def plot_compare(d1, d2, bs, ep, out):
    from matplotlib.lines import Line2D
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    # 配色 = init（橘=random / 藍=imagenet）；線型 = Top-1 實線 / Top-5 虛線
    # (a) loss overlay
    axL.plot(d1["ep"], d1["loss"], color=C_T1, linewidth=3, label="Random Init")
    axL.plot(d2["ep"], d2["loss"], color=C_T2, linewidth=3, label="ImageNet Init")
    axL.set_ylabel("InfoNCE Loss", fontsize=FONT_LABEL, labelpad=8)
    axL.set_title("(a) Contrastive Loss", fontsize=FONT_TITLE, pad=10)
    axL.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    # (b) accuracy overlay；Top-5 用較細的長虛線（線太粗時 legend 裡縫會被填滿、看起來像實線）
    DASH = (0, (6, 4))   # 長虛線、縫也大
    axR.plot(d1["ep"], d1["top1"], color=C_T1, linewidth=3)
    axR.plot(d1["ep"], d1["top5"], color=C_T1, linewidth=2.2, linestyle=DASH)
    axR.plot(d2["ep"], d2["top1"], color=C_T2, linewidth=3)
    axR.plot(d2["ep"], d2["top5"], color=C_T2, linewidth=2.2, linestyle=DASH)
    axR.set_ylabel("Contrastive Accuracy (%)", fontsize=FONT_LABEL, labelpad=8)
    axR.set_title("(b) Instance-Discrimination Accuracy", fontsize=FONT_TITLE, pad=10)
    # 兩組獨立 legend：顏色區分 init、線型區分 Top-1/Top-5（最簡潔）
    init_handles = [Line2D([0], [0], color=C_T1, lw=3, label="Random Init"),
                    Line2D([0], [0], color=C_T2, lw=3, label="ImageNet Init")]
    metric_handles = [Line2D([0], [0], color="0.3", lw=3, ls="-", label="Top-1"),
                      Line2D([0], [0], color="0.3", lw=2.2, ls=DASH, label="Top-5")]
    # 兩個 legend 在右下角上下緊貼：Top-1/Top-5 在上、Random/ImageNet 在下
    leg1 = axR.legend(handles=init_handles, fontsize=FONT_LEGEND, framealpha=0.9,
                      loc="lower right", bbox_to_anchor=(1.0, 0.0))
    axR.add_artist(leg1)
    axR.legend(handles=metric_handles, fontsize=FONT_LEGEND, framealpha=0.9,
               loc="lower right", bbox_to_anchor=(1.0, 0.155))

    for ax in (axL, axR):
        ax.set_xlabel("Pretraining Epoch", fontsize=FONT_LABEL, labelpad=8)
        style_ax(ax)
    _save(fig, out)


def plot_sweep_bs(datas, bs_list, out):
    """同一 init、固定 ep，疊不同 batch size。左=loss，右=Top-1 only。色由 viridis 依 bs 排序。"""
    colors = [plt.cm.viridis(x) for x in (0.0, 0.28, 0.5, 0.72, 0.95)]
    colors = colors[:len(bs_list)] if len(bs_list) <= 5 else \
        [plt.cm.viridis(i / (len(bs_list) - 1)) for i in range(len(bs_list))]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    for d, bs, c in zip(datas, bs_list, colors):
        axL.plot(d["ep"], d["loss"], color=c, linewidth=3, label=f"bs = {bs}")
        axR.plot(d["ep"], d["top1"], color=c, linewidth=3, label=f"bs = {bs}")
    axL.set_ylabel("InfoNCE Loss", fontsize=FONT_LABEL, labelpad=8)
    axL.set_title("(a) Contrastive Loss", fontsize=FONT_TITLE, pad=10)
    axR.set_ylabel("Contrastive Top-1 Accuracy (%)", fontsize=FONT_LABEL, labelpad=8)
    axR.set_title("(b) Instance-Discrimination Accuracy", fontsize=FONT_TITLE, pad=10)
    axL.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    for ax in (axL, axR):
        ax.set_xlabel("Pretraining Epoch", fontsize=FONT_LABEL, labelpad=8)
        style_ax(ax)
    _save(fig, out)


def _save(fig, out):
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="overlay θ¹(random init) vs θ²(ImageNet init)")
    ap.add_argument("--sweep_bs", action="store_true",
                    help="同一 init、固定 ep，疊不同 batch size（右圖只畫 Top-1）")
    ap.add_argument("--bs_list", nargs="+", default=["16", "32", "64", "128", "256"],
                    help="sweep_bs 模式的 batch size 清單")
    ap.add_argument("--init", default="theta2",
                    choices=["theta1", "theta2", "random", "imagenet"],
                    help="單一/sweep_bs 模式用：哪一種 init（預設 theta2）")
    ap.add_argument("--bs", default=None, help="單一/compare 模式必填")
    ap.add_argument("--ep", required=True)
    ap.add_argument("--lr", default="0.0002", help="單一模式的 SimCLR pretraining lr（需與檔名一致）")
    ap.add_argument("--lr1", default="0.0002", help="compare 模式：θ¹ 的 pretraining lr")
    ap.add_argument("--lr2", default="0.0002", help="compare 模式：θ² 的 pretraining lr")
    ap.add_argument("--json_dir", default=os.path.join(ROOT, "SSL", "simclr", "json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    figdir = os.path.join(os.path.dirname(__file__), "figs")

    if args.sweep_bs:
        datas = [load_history(args.init, args.lr, bs, args.ep, args.json_dir) for bs in args.bs_list]
        out = args.out or os.path.join(figdir, f"simclr_curve_bssweep_{args.init}_ep{args.ep}.png")
        plot_sweep_bs(datas, args.bs_list, out)
        return

    if not args.bs:
        ap.error("單一/compare 模式需要 --bs")

    if args.compare:
        d1 = load_history("theta1", args.lr1, args.bs, args.ep, args.json_dir)
        d2 = load_history("theta2", args.lr2, args.bs, args.ep, args.json_dir)
        n1, n2 = len(d1["ep"]), len(d2["ep"])
        out = args.out or os.path.join(figdir, f"simclr_curve_compare_bs{args.bs}_ep{args.ep}.png")
        plot_compare(d1, d2, args.bs, args.ep, out)
        if n1 < int(args.ep) or n2 < int(args.ep):
            print(f"⚠️ partial：θ¹={n1}/{args.ep} ep, θ²={n2}/{args.ep} ep。θ¹ 跑完後請重跑本指令產生正式圖。")
    else:
        d = load_history(args.init, args.lr, args.bs, args.ep, args.json_dir)
        out = args.out or os.path.join(
            figdir, f"simclr_curve_{args.init}_bs{args.bs}_ep{args.ep}_lr{args.lr}.png")
        plot_single(d, args.bs, args.ep, out)
        print(f"final loss={d['loss'][-1]:.4f}  top1={d['top1'][-1]:.2f}%  top5={d['top5'][-1]:.2f}%")


if __name__ == "__main__":
    main()
