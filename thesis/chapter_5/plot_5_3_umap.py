#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3 質性分析：特徵空間 UMAP 視覺化
=====================================

用某個 finetuned 模型的 **512 維 last-layer 特徵**（global-avgpool 輸出，fc 之前），
把整個 train set（2032 張）投影到 2D（UMAP）。每個 ground-truth 類別用不同顏色+marker；
再把某個 AL 方法在某 portion 「累積選取」的影像，用**紅色空心方框**框起來。

模型來源：thesis/gradcam/ckpt/（與 gradcam 共用）。default = simclr_p100_4x.pth
（θ²-SimCLR、aug4、100% finetune；由 finetune_full_model.py 產生）。
也可指向既有 imagenet_p100_4x.pth（ImageNet-init）。載入時自動分辨 plain resnet18 / ResNetSimCLR。

AL 選樣來源：AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json 的 cumulative（default seed=42）。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_umap.py --device cuda:6           # default: 全7法 × {30,15}%
    python3 thesis/chapter_5/plot_5_3_umap.py --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from classification.model.resnet import get_resnet18_classifier          # noqa: E402
from classification.model.simclr.resnet_simclr import ResNetSimCLR       # noqa: E402
from torchvision import datasets, transforms                             # noqa: E402

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
CKPT_DIR = os.path.join(REPO, "thesis", "gradcam", "ckpt")
LABELED_IDS_DIR = os.path.join(REPO, "classification", "exp_results",
                               "classification_hard", "AL_simclr", "labeled_ids")
OUT_DIR = os.path.join(REPO, "thesis", "chapter_5", "figs")
CACHE_DIR = os.path.join(REPO, "thesis", "chapter_5", "umap_cache")

# 顯示名稱（與 5.3 其它圖一致）
NAME_ABBR = {"Seborrhoeic keratosis": "SK", "Solar lentigo": "SL"}
# 7 類各自顏色 + marker（ImageFolder 字母序：Eczema,Nevus,Normal,Psoriasis,SK,SL,Vitiligo）
CLASS_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF"]
CLASS_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

STRATEGY_LABEL = {
    "conf": "Confidence", "margin": "Margin", "entropy": "Entropy",
    "coreset": "Core-set", "typiclust": "TypiClust",
    "badge": "BADGE", "cluster_margin": "Cluster-Margin",
}


# --------------------------------------------------------------------------- #
def build_model(ckpt_path, num_classes=7):
    """讀 state_dict 建特徵抽取器，自動分辨三種來源：
       (a) plain resnet18 finetuned .pth（imagenet-init）
       (b) ResNetSimCLR finetuned .pth（backbone.fc = Linear(512,7)）
       (c) frozen SimCLR backbone .pkl（backbone.fc = MLP projection head）
    用 strict=False 載入：fc 不論是哪種都會被丟掉（我們只取 avgpool 後 512 維），
    conv/bn/layer 權重照常載入。"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith(("conv1", "backbone")) for k in sd):
        sd = sd["state_dict"]
    is_simclr = any(k.startswith("backbone.") for k in sd)
    if is_simclr:
        model = ResNetSimCLR("resnet18", 32)        # backbone.fc 預設是 MLP projection head
        backbone = model.backbone
    else:
        model = get_resnet18_classifier(num_classes=num_classes, pretrained=False)
        backbone = model
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 只允許 fc 相關的不吻合（特徵抽取不需要 fc）
    bad = [k for k in unexpected if ".fc" not in k] + [k for k in missing if ".fc" not in k]
    if bad:
        print(f"[warn] unexpected non-fc key mismatch (first few): {bad[:5]}")
    feat_net = nn.Sequential(*list(backbone.children())[:-1])   # 去掉最後 fc → 512 維
    return feat_net.eval(), is_simclr


@torch.no_grad()
def extract_features(feat_net, device, batch_size=64):
    """對整個 train set 抽 512 維特徵，順序 = ImageFolder.samples（= labeled_ids index 空間）。"""
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feat_net.to(device)
    feats, labels = [], []
    for x, y in loader:
        f = feat_net(x.to(device)).flatten(1)        # (B,512)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels), ds.classes


def get_embedding(ckpt_path, device, method="umap", recompute=False):
    """抽 512 維特徵 + 降到 2D（method='umap' 或 'tsne'）；
    以 (ckpt, method) 快取（同模型同方法只算一次，跨 strategy/portion 重用）。"""
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{stem}_{method}.npz")
    if os.path.exists(cache) and not recompute:
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}")
        return d["emb"], d["labels"], list(d["classes"])
    feat_net, is_simclr = build_model(ckpt_path)
    print(f"model: {'ResNetSimCLR' if is_simclr else 'plain resnet18'}  ({stem})")
    X, y, classes = extract_features(feat_net, device)
    print(f"features: {X.shape}; running {method.upper()} ...")
    if method == "umap":
        import umap
        emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=42).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        # 慣例：先 PCA→50 再 t-SNE，較快且較穩
        Xp = PCA(n_components=50, random_state=42).fit_transform(X) if X.shape[1] > 50 else X
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   learning_rate="auto", random_state=42).fit_transform(Xp)
    else:
        raise ValueError(method)
    np.savez(cache, emb=emb, labels=y, classes=np.array(classes, dtype=object))
    print(f"[saved cache] {cache}")
    return emb, y, classes


def plot_base(emb, labels, classes, method, out_path):
    """純特徵空間：只有類別著色、無任何選取標註。"""
    disp = [NAME_ABBR.get(c, c) for c in classes]
    n = len(classes)
    counts = np.array([int((labels == ci).sum()) for ci in range(n)])
    order = list(np.argsort(counts)[::-1])
    fig, ax = plt.subplots(figsize=(10, 9))
    for ci in range(n):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=90, c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
                   alpha=0.7, linewidths=0, zorder=2)
    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=10, label=disp[ci]) for ci in order]
    ax.legend(handles=class_h, fontsize=13, framealpha=0.95, loc="best")
    axname = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_xlabel(f"{axname}-1", fontsize=20, labelpad=8)
    ax.set_ylabel(f"{axname}-2", fontsize=20, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def load_selected(strategy, portion, seed, mode="cumulative"):
    path = os.path.join(LABELED_IDS_DIR, f"{strategy}_seed{seed}_bs16.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    pk = str(float(portion))
    if pk not in d:
        return None
    return d[pk][mode]


def plot_umap(emb, labels, classes, selected_idx, strategy, portion, seed, out_path,
              highlight="star", method="umap"):
    """highlight:
       'star' = 顏色表類別；未選=小圓點(淡)、已選=大星號(實心+黑邊)。       [預設，較清楚]
       'box'  = 顏色+marker 表類別；已選=紅色空心方框（舊樣式）。"""
    disp = [NAME_ABBR.get(c, c) for c in classes]
    n = len(classes)
    sel = np.zeros(len(labels), dtype=bool)
    if selected_idx:
        sel[np.array(selected_idx)] = True
    # legend 由上到下依「該類別張數」由多到少
    counts = np.array([int((labels == ci).sum()) for ci in range(n)])
    order = list(np.argsort(counts)[::-1])
    fig, ax = plt.subplots(figsize=(10, 9))

    if highlight == "star":
        # 背景點與選取點「一樣大」，只靠形狀(圓/星)+透明度/黑邊區分，使選取比例視覺上不被放大
        UNSEL_SIZE, SEL_SIZE = 60, 90   # 星號同 s 視覺較小，故略大一點抵銷
        for ci in range(n):
            cls = labels == ci
            m_un = cls & ~sel
            ax.scatter(emb[m_un, 0], emb[m_un, 1], s=UNSEL_SIZE, c=CLASS_COLORS[ci], marker="o",
                       alpha=0.40, linewidths=0, zorder=2)
        for ci in range(n):
            m_se = (labels == ci) & sel
            ax.scatter(emb[m_se, 0], emb[m_se, 1], s=SEL_SIZE, c=CLASS_COLORS[ci], marker="*",
                       alpha=0.95, edgecolors="black", linewidths=0.5, zorder=4)
        # legend：類別色（圓）+ 選取狀態（星/圓，灰）；類別依張數多→少
        class_h = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[ci],
                          markeredgecolor="none", markersize=9, label=disp[ci]) for ci in order]
        status_h = [
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="0.25",
                   markeredgecolor="black", markersize=14, label=f"Selected (n={int(sel.sum())})"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.6",
                   markeredgecolor="none", markersize=9, label="Not selected"),
        ]
        ax.legend(handles=class_h + status_h, fontsize=13, framealpha=0.95, loc="best", ncol=1)
    else:  # box：所有點大小一致（=star 大小），選取者額外加黑色空心方框
        PT_SIZE = 90
        for ci in range(n):
            m = labels == ci
            ax.scatter(emb[m, 0], emb[m, 1], s=PT_SIZE, c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
                       alpha=0.65, linewidths=0, zorder=2)
        if sel.any():
            ax.scatter(emb[sel, 0], emb[sel, 1], s=240, facecolors="none", edgecolors="black",
                       linewidths=1.5, marker="s", zorder=4)
        # legend：類別（色+marker）依張數多→少，最後接 Selected
        class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                          markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                          markersize=10, label=disp[ci]) for ci in order]
        sel_h = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
                        markeredgecolor="black", markersize=12,
                        label=f"Selected (n={int(sel.sum())})")]
        ax.legend(handles=class_h + sel_h, fontsize=13, framealpha=0.95, loc="best")

    axname = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_xlabel(f"{axname}-1", fontsize=20, labelpad=8)
    ax.set_ylabel(f"{axname}-2", fontsize=20, labelpad=8)
    ax.set_title(f"{STRATEGY_LABEL.get(strategy, strategy)}   (ρ={portion:g}%, seed{seed})",
                 fontsize=22, pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "simclr_p100_4x.pth"),
                    help="模型 .pth（thesis/gradcam/ckpt 下）；default = simclr_p100_4x.pth")
    ap.add_argument("--strategy", nargs="+",
                    default=["margin", "coreset", "cluster_margin"])
    ap.add_argument("--portion", nargs="+", type=float, default=[30.0, 15.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["cumulative", "selected"], default="cumulative")
    ap.add_argument("--highlight", choices=["star", "box"], default="star",
                    help="star=選取用星號/未選小圓點(預設) ; box=舊的紅色方框")
    ap.add_argument("--method", choices=["umap", "tsne"], default="umap",
                    help="2D 降維方法（圖與快取依方法分開）")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--base", action="store_true",
                    help="只畫純特徵空間（類別著色、無任何選取標註），不跑各 AL")
    ap.add_argument("--recompute", action="store_true", help="強制重算 UMAP（忽略快取）")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"[error] ckpt 不存在: {args.ckpt}\n（先跑 finetune_full_model.py 產生，或改 --ckpt 指向 imagenet_p100_4x.pth）")
        return

    emb, labels, classes = get_embedding(args.ckpt, args.device, method=args.method,
                                         recompute=args.recompute)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]

    for strat in args.strategy:
        for p in args.portion:
            sel = load_selected(strat, p, args.seed, args.mode)
            if sel is None:
                print(f"[skip] {strat} @ {p}% (no labeled_ids)")
                continue
            # 依 method / feature-extractor 模型分子資料夾。box 不加後綴、star 加 _star
            suffix = "" if args.highlight == "box" else f"_{args.highlight}"
            out = os.path.join(args.out_dir, args.method, stem,
                               f"5_3_{args.method}_{strat}_p{p:g}_seed{args.seed}_{args.mode}{suffix}.png")
            plot_umap(emb, labels, classes, sel, strat, p, args.seed, out,
                      highlight=args.highlight, method=args.method)


if __name__ == "__main__":
    main()
