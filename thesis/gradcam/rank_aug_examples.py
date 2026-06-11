#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
為 GradCAM cherry-picking 用：對 val 裡 7 類「所有」影像做一次完整 inference，
對每個 baseline 算
    delta = P_aug4(ground-truth) − P_base(ground-truth)
（aug4 模型 vs baseline 模型，對「該圖真實類別」的機率差），**依類別**由大到小排序，
印出每個 label 中「aug4 比該 baseline 幫助最大」的 top-5 影像。

一次可給多個 baseline（--base label:ckpt），常見為 no-aug 與 horizontal(HF)，
→ 同一次 inference 就同時得到「aug4 vs no-aug」和「aug4 vs HF」兩份排序。

重用 gradcam_view.py 的 model 載入（自動分辨 plain resnet / SimCLR）與 eval transform。

用法（repo 根執行）：
  python3 thesis/gradcam/rank_aug_examples.py \
      --ckpt_aug4 thesis/gradcam/ckpt/imagenet_p100_4x.pth \
      --base noaug:thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth \
      --base HF:thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth \
      --device cuda:0 --out thesis/gradcam/out/rank_aug4.csv
"""
import os, sys, csv, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from gradcam_view import build_model, _TF, NUM_CLASSES, DATA_DIR, _ROOT


@torch.no_grad()
def probs(model, x):
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_aug4", required=True, help="aug4(HF+VF+HVF) 模型 ckpt")
    ap.add_argument("--base", action="append", required=True, metavar="LABEL:CKPT",
                    help="baseline，格式 label:ckpt路徑；可重複（如 noaug:..  HF:..）")
    ap.add_argument("--task_type", default="hard", choices=list(NUM_CLASSES))
    ap.add_argument("--split", default="val", help="掃哪個資料夾（GradCAM 用的同一份，預設 val）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=5, help="每類別印前幾名（預設 5）")
    ap.add_argument("--out", default=None, help="可選：完整排序存成 CSV（含各 baseline 的 Δ）")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    nc = NUM_CLASSES[args.task_type]

    bases = []                                  # [(label, model), ...]
    for spec in args.base:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"--base 格式應為 label:ckpt，收到：{spec}")
        m, _, sc = build_model(path, nc, device)
        bases.append((label, m))
        print(f"baseline '{label}' = {'SimCLR' if sc else 'plain resnet'}  ({path})")
    m4, _, sc4 = build_model(args.ckpt_aug4, nc, device)
    print(f"aug4 = {'SimCLR' if sc4 else 'plain resnet'}  ({args.ckpt_aug4})")

    root = os.path.join(_ROOT, "ds", "classification", DATA_DIR[args.task_type], args.split)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    # 一次 full inference，記下 aug4 與每個 baseline 對 ground-truth 的機率
    rows, n_total = [], 0
    for ci, cname in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, cname))):
            path = os.path.join(root, cname, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = _TF(img).unsqueeze(0).to(device)
            p4 = probs(m4, x)
            row = {"class": cname, "file": fn, "image": path,
                   "p_aug4": float(p4[ci]), "pred_aug4": classes[int(p4.argmax())]}
            for label, mb in bases:
                pb = probs(mb, x)
                row[f"p_{label}"] = float(pb[ci])
                row[f"delta_{label}"] = float(p4[ci] - pb[ci])
                row[f"pred_{label}"] = classes[int(pb.argmax())]
            rows.append(row)
            n_total += 1
    print(f"\n完整 inference 共 {n_total} 張（{len(classes)} 類）。\n")

    def show(rs, label):
        pk, dk = f"p_{label}", f"delta_{label}"
        print(f"  {'file':<28} {'P_'+label:>9} {'P_aug4':>8} {'Δ':>8}   pred:{label}→aug4")
        print("  " + "-" * 90)
        for r in rs:
            chg = "same" if r[f"pred_{label}"] == r["pred_aug4"] else f"{r[f'pred_{label}']}→{r['pred_aug4']}"
            print(f"  {r['file']:<28} {r[pk]:>9.3f} {r['p_aug4']:>8.3f} {r[dk]:>+8.3f}   {chg}")

    # 每個 baseline 各印一份：依類別 top-k
    for label, _ in bases:
        dk = f"delta_{label}"
        print("\n" + "#" * 100)
        print(f"#  aug4  vs  {label}      Δ = P_aug4(gt) − P_{label}(gt)，依類別 top-{args.topk}")
        print("#" * 100)
        for cname in classes:
            rs = sorted([r for r in rows if r["class"] == cname], key=lambda r: r[dk], reverse=True)
            print(f"\n■ {cname}  （n={len(rs)}）")
            show(rs[:args.topk], label)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cols = ["class", "file", "image", "p_aug4", "pred_aug4"]
        for label, _ in bases:
            cols += [f"p_{label}", f"delta_{label}", f"pred_{label}"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n完整資料({len(rows)} 張)已存 -> {args.out}")


if __name__ == "__main__":
    main()
