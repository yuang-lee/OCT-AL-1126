#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GradCAM viewer — 對「某個 fine-tuned ckpt」在「某張 testing 影像」上畫 Grad-CAM / Grad-CAM++。

與學長 notebook 一致：target layer = ResNet-18 的 **layer4 最後一個 block**
（plain resnet → model.layer4[-1]；SimCLR 包裝 → model.backbone.layer4[-1]，自動偵測）。
前處理沿用本 repo 的 eval transform（ToTensor + Normalize([0.5]*3,[0.5]*3)，不 resize）。
自帶實作、不需安裝 pytorch_grad_cam。

用法（repo 根執行）：
  python3 thesis/gradcam/gradcam_view.py \
      --ckpt path/to/model.pth \
      --image ds/classification/seven_class/val/<Class>/<img>.png \
      --task_type hard --method gradcam++ --device cuda:0 \
      --out thesis/gradcam/out/example1.png
  # --target_class 預設用模型預測的類別；可指定整數對某類算 CAM。
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

# 從 repo 根 import
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR

NUM_CLASSES = {"easy": 2, "medium": 4, "hard": 7}
DATA_DIR = {"easy": "two_class", "medium": "four_class", "hard": "seven_class"}

# 與 repo eval 一致
_TF = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.5] * 3, [0.5] * 3)])


def build_model(ckpt_path, num_classes, device):
    """讀 state_dict，自動分辨 plain resnet18 / ResNetSimCLR，回傳 (model, target_layer, is_simclr)。"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith(("conv1", "backbone")) for k in sd):
        sd = sd["state_dict"]
    is_simclr = any(k.startswith("backbone.") for k in sd)
    if is_simclr:
        model = ResNetSimCLR("resnet18", 32)
        in_f = model.backbone.fc[0].in_features
        model.backbone.fc = nn.Linear(in_f, num_classes)
        target_layer = model.backbone.layer4[-1]
    else:
        model = get_resnet18_classifier(num_classes=num_classes, pretrained=False)
        target_layer = model.layer4[-1]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, target_layer, is_simclr


def compute_cam(model, target_layer, x, cls, method="gradcam++"):
    """對 x（[1,3,H,W]）算 target layer 的 CAM；回傳 [h,w] numpy（未上採樣、未正規化）。"""
    store = {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: store.update(A=o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: store.update(G=go[0]))
    try:
        logits = model(x)
        if cls < 0:
            cls = int(logits.argmax(1).item())
        model.zero_grad(set_to_none=True)
        logits[0, cls].backward()
        A = store["A"][0]                 # [C,h,w]
        G = store["G"][0]                 # [C,h,w]
        if method == "gradcam":
            w = G.mean(dim=(1, 2))                                    # [C]
        else:  # grad-cam++
            G2, G3 = G ** 2, G ** 3
            denom = 2 * G2 + (A * G3).sum(dim=(1, 2), keepdim=True)
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = G2 / denom
            w = (alpha * torch.relu(G)).sum(dim=(1, 2))               # [C]
        cam = torch.relu((w[:, None, None] * A).sum(0))              # [h,w]
        return cam.detach(), cls, torch.softmax(logits, dim=1)[0].detach()
    finally:
        h1.remove(); h2.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned model 的 state_dict 路徑")
    ap.add_argument("--image", required=True, help="一張 testing 影像路徑")
    ap.add_argument("--task_type", default="hard", choices=list(NUM_CLASSES))
    ap.add_argument("--target_class", type=int, default=-1, help="對哪個類別算 CAM；-1=用模型預測類別")
    ap.add_argument("--method", default="gradcam++", choices=["gradcam", "gradcam++"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay 熱圖透明度")
    ap.add_argument("--out", default=None, help="輸出 PNG（預設 thesis/gradcam/out/<image檔名>_cam.png）")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    num_classes = NUM_CLASSES[args.task_type]
    model, target_layer, is_simclr = build_model(args.ckpt, num_classes, device)
    print(f"model = {'ResNetSimCLR(.backbone)' if is_simclr else 'plain resnet18'}; "
          f"target layer = {'backbone.' if is_simclr else ''}layer4[-1]")

    # class 名稱（資料夾名，ImageFolder 排序）
    train_dir = os.path.join(_ROOT, "ds", "classification", DATA_DIR[args.task_type], "train")
    class_names = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else [str(i) for i in range(num_classes)]

    img = Image.open(args.image).convert("RGB")
    rgb = np.array(img).astype(np.float32) / 255.0       # [H,W,3]
    H, W = rgb.shape[:2]
    x = _TF(img).unsqueeze(0).to(device)

    cam, cls, probs = compute_cam(model, target_layer, x, args.target_class, args.method)
    # 上採樣到原圖大小 + 正規化 0~1
    cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()

    heat = cm.jet(cam)[..., :3]                          # [H,W,3]
    overlay = np.clip((1 - args.alpha) * rgb + args.alpha * heat, 0, 1)

    cname = class_names[cls] if cls < len(class_names) else str(cls)
    print(f"預測/目標類別 = {cls} ({cname})，p={probs[cls].item():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb);     axes[0].set_title("Input", fontsize=14)
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title(args.method, fontsize=14)
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay — {cname} (p={probs[cls]:.2f})", fontsize=14)
    for a in axes:
        a.axis("off")
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(__file__), "out",
                                   os.path.splitext(os.path.basename(args.image))[0] + "_cam.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
