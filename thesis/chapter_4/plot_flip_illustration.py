#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4x 資料增強（翻轉）示意圖：挑一張 classification testing 影像，畫 2×2 四宮格——
  左上 (a) Original image      右上 (b) Horizontal Flip (HF)
  左下 (c) Vertical Flip (VF)  右下 (d) Horizontal & Vertical Flip (HVF)
形成一張大圖（含各子圖 caption），風格對齊碩論（Arial、dpi 300、白底）。

HVF = 先水平、再垂直翻轉（等同 180° 旋轉）。

用法（repo 根執行）：
  python3 thesis/chapter_4/plot_flip_illustration.py \
      --image 'ds/classification/seven_class/val/Eczema/20220307_102605B.png' \
      --gray --out thesis/chapter_4/figs/flip_illustration.png
"""
import os, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.5,
})
CAP = 22   # 子圖 caption 字級（對齊碩論其他圖）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="一張 testing 影像路徑")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figs", "flip_illustration.png"))
    ap.add_argument("--gray", action="store_true", help="以灰階顯示（OCT 影像建議加）")
    ap.add_argument("--title", default=None, help="可選：整體標題（一般留給 LaTeX caption）")
    args = ap.parse_args()

    img = Image.open(args.image).convert("L" if args.gray else "RGB")
    HF = img.transpose(Image.FLIP_LEFT_RIGHT)
    VF = img.transpose(Image.FLIP_TOP_BOTTOM)
    HVF = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    panels = [
        (img, "(a) Original image"),
        (HF,  "(b) Horizontal Flip (HF)"),
        (VF,  "(c) Vertical Flip (VF)"),
        (HVF, "(d) Horizontal & Vertical Flip (HVF)"),
    ]

    W, H = img.size
    pw = 4.0 * (W / H)            # 每格寬度依影像長寬比，避免變形
    fig, axes = plt.subplots(2, 2, figsize=(2 * pw, 2 * 4.0 + 0.8), constrained_layout=True)
    cmap = "gray" if args.gray else None
    for ax, (im, cap) in zip(axes.flat, panels):
        ax.imshow(np.asarray(im), cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():       # 每格留淡灰細框
            s.set_linewidth(1.2); s.set_color("0.4")
        ax.set_xlabel(cap, fontsize=CAP, labelpad=10)

    if args.title:
        fig.suptitle(args.title, fontsize=CAP + 4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
