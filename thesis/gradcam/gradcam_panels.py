#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.2 GradCAM 對比圖總指揮：
 1) 對 val 7 類「所有」影像做 full inference（noaug / HF / VF / aug4 四模型），
    每類別挑「aug4 比 noaug 提升最多」與「aug4 比 HF 提升最多」各 top-5（且 aug4 機率較高）。
 2) 對每張選中影像：四種 aug 各畫一張 GradCAM 存到 out/<class>/<imgid>/<aug>_cam.png。
 3) 綜合 2×2 圖（碩論 Arial/dpi300 風格），存 out/<class>/<imgid>/panel.png：
      (a) Original  [title=Ground-truth 類別簡寫]
      (b) w/o Aug GradCAM      [title=P=..]
      (c) HF GradCAM           [title=P=..]
      (d) HF+VF+HVF GradCAM    [title=P=..]
    （P = 模型對「該圖真實類別」的 softmax 機率；GradCAM 也針對真實類別。）

用法（repo 根）：
  python3 thesis/gradcam/gradcam_panels.py --device cuda:8
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(__file__))
from gradcam_view import build_model, compute_cam, _TF, NUM_CLASSES, DATA_DIR, _ROOT

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.5,
})

# 疾病簡寫 — follow 碩論（thesis/plot/class_dist.py）
ABBR = {
    "Normal": "Healthy", "Eczema": "Eczema", "Psoriasis": "Psoriasis",
    "Solar lentigo": "SL", "Nevus": "Nevus",
    "Seborrhoeic keratosis": "SK", "Vitiligo": "Vitiligo",
}
TITLE = 22


def cam_overlay(model, target_layer, img_pil, cls, device, alpha=0.5):
    """回傳 (overlay RGB[0..1], P(cls))；GradCAM++ 針對 cls。"""
    x = _TF(img_pil.convert("RGB")).unsqueeze(0).to(device)
    cam, _, probs = compute_cam(model, target_layer, x, cls, "gradcam++")
    H, W = np.asarray(img_pil).shape[:2]
    cam = F.interpolate(cam[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    rgb = np.asarray(img_pil.convert("RGB")).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)
    return overlay, float(probs[cls])


def save_single(arr01, path):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)


def build_panel(img_pil, overlays, probs, abbr, out):
    """2×2：原圖 + noaug/HF/aug4 三張 GradCAM。"""
    gray = np.asarray(img_pil.convert("L"))
    W, H = img_pil.size
    r = W / H
    cells = [
        (f"(a) Original — {abbr}", gray, "gray", None),
        ("(b) w/o Aug",            overlays["noaug"], None, probs["noaug"]),
        ("(c) HF",                 overlays["HF"],    None, probs["HF"]),
        ("(d) HF+VF+HVF",          overlays["aug4"],  None, probs["aug4"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(2 * 4.0 * r, 2 * 4.0 + 0.9), constrained_layout=True)
    for ax, (lab, im, cmap, p) in zip(axes.flat, cells):
        ax.imshow(im, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.2); s.set_color("0.4")
        ax.set_title(lab if p is None else f"{lab}    P = {p:.2f}", fontsize=TITLE, pad=8)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def _generate(selected, log, models, abbr, out_root, device):
    """對 selected 的每張影像：四 aug GradCAM + 綜合 panel；log 非空才寫 CSV。"""
    for i, ((cname, fn), r) in enumerate(selected.items(), 1):
        img = Image.open(r["path"]).convert("RGB")
        imgid = os.path.splitext(fn)[0]
        outdir = os.path.join(out_root, cname, imgid)
        os.makedirs(outdir, exist_ok=True)
        overlays, probs = {}, {}
        for name, (m, tl) in models.items():
            ov, pg = cam_overlay(m, tl, img, r["ci"], device)
            overlays[name], probs[name] = ov, pg
            save_single(ov, os.path.join(outdir, f"{name}_cam.png"))
        save_single(np.asarray(img).astype(np.float32) / 255.0, os.path.join(outdir, "original.png"))
        build_panel(img, overlays, probs, abbr.get(cname, cname), os.path.join(outdir, "panel.png"))
        print(f"  [{i}/{len(selected)}] {cname}/{imgid}  "
              f"P: noaug={probs['noaug']:.2f} HF={probs['HF']:.2f} aug4={probs['aug4']:.2f}")
    if log:
        csv_path = os.path.join(out_root, "panels_selected.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["class", "file", "cmp", "delta",
                                              "P_noaug", "P_HF", "P_VF", "P_aug4"])
            w.writeheader(); w.writerows(log)
        print(f"\n選圖清單 -> {csv_path}")
    print(f"每張結果在 {out_root}/<class>/<imgid>/（panel.png + 四 aug _cam.png + original.png）")


def main():
    ap = argparse.ArgumentParser()
    ck = "thesis/gradcam/ckpt"
    ap.add_argument("--ckpt_noaug", default=f"{ck}/imagenet_p100_1x_noaug.pth")
    ap.add_argument("--ckpt_hf", default=f"{ck}/imagenet_p100_2x_hf.pth")
    ap.add_argument("--ckpt_vf", default=f"{ck}/imagenet_p100_2x_vf.pth")
    ap.add_argument("--ckpt_aug4", default=f"{ck}/imagenet_p100_4x.pth")
    ap.add_argument("--task_type", default="hard", choices=list(NUM_CLASSES))
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out_root", default="thesis/gradcam/out")
    ap.add_argument("--images", nargs="+", default=None,
                    help="指定影像路徑（給定則跳過排序，直接對這些圖做 panel；類別取自上層資料夾名）")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    nc = NUM_CLASSES[args.task_type]
    models = {}
    for name, c in [("noaug", args.ckpt_noaug), ("HF", args.ckpt_hf),
                    ("VF", args.ckpt_vf), ("aug4", args.ckpt_aug4)]:
        m, tl, _ = build_model(c, nc, device)
        models[name] = (m, tl)
    print("4 個模型載入完成：", list(models))

    root = os.path.join(_ROOT, "ds", "classification", DATA_DIR[args.task_type], args.split)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    # 指定影像模式：跳過排序，直接對給定影像做 panel（類別取自上層資料夾名）
    if args.images:
        selected, log = {}, []
        for path in args.images:
            cname = os.path.basename(os.path.dirname(path))
            if cname not in classes:
                print(f"  [skip] {path}：類別 '{cname}' 不在 {classes}"); continue
            fn = os.path.basename(path)
            selected[(cname, fn)] = {"class": cname, "ci": classes.index(cname),
                                     "file": fn, "path": path, "p": {}}
        print(f"指定影像模式：{len(selected)} 張")
        _generate(selected, log, models, ABBR, args.out_root, device)
        return

    # 1) full inference：每張影像對 4 模型在「真實類別」的機率
    recs = []
    for ci, cname in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, cname))):
            path = os.path.join(root, cname, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = _TF(img).unsqueeze(0).to(device)
            with torch.no_grad():
                p = {name: float(torch.softmax(m(x), 1)[0, ci].cpu()) for name, (m, _) in models.items()}
            recs.append({"class": cname, "ci": ci, "file": fn, "path": path, "p": p})
    print(f"full inference 完成：{len(recs)} 張")

    # 2) 每類別挑 top-5（aug4−noaug）與 top-5（aug4−HF），aug4 機率較高
    selected, log = {}, []
    for cname in classes:
        sub = [r for r in recs if r["class"] == cname]
        for base in ["noaug", "HF"]:
            ranked = sorted(sub, key=lambda r: r["p"]["aug4"] - r["p"][base], reverse=True)
            for r in ranked[:args.topk]:
                d = r["p"]["aug4"] - r["p"][base]
                if d <= 0:
                    continue
                selected.setdefault((cname, r["file"]), r)
                log.append({"class": cname, "file": r["file"], "cmp": f"aug4-{base}",
                            "delta": round(d, 4), **{f"P_{k}": round(v, 4) for k, v in r["p"].items()}})
    print(f"選中 {len(selected)} 張（{len(log)} 條 top-5 紀錄）")

    # 3) 對每張：四 aug GradCAM + 綜合圖
    _generate(selected, log, models, ABBR, args.out_root, device)


if __name__ == "__main__":
    main()
