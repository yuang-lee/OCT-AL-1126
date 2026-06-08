"""
Coreset / K-Center Greedy (Sener & Savarese, ICLR 2018) — 正確版。

原論文：在 penultimate-layer 特徵空間做 **furthest-first traversal，conditioned on
所有「已標註」樣本**——每輪挑「離最近的已標註中心最遠」的未標註點，加入後更新距離。

對 diversity_wrong.py 的修正：原本只收到未標註集、從隨機未標註點開始，完全沒看
已標註資料（等於在未標註池重做 k-center），會選到已標註點附近的冗餘。此處讓函式
也吃 `labeled_idx`，把 min_distances 初始化成「每個未標註點到最近已標註點的距離」。
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _extract_features(feature_extractor, full_dataset, idx_list, device, desc):
    """抽 penultimate-layer 特徵，回傳 numpy [len(idx_list), D]。"""
    if len(idx_list) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    loader = DataLoader(Subset(full_dataset, idx_list), batch_size=128,
                        shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    feats = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc=desc):
            inputs = inputs.to(device, non_blocking=True)
            f = feature_extractor(inputs)
            feats.append(f.view(f.size(0), -1).cpu().numpy())
    return np.vstack(feats).astype(np.float32)


def coreset(model, data_dir, unlabel_data_idx, num_data_to_label, device, labeled_idx=None):
    """
    Args:
        labeled_idx: 目前「已標註」樣本的全資料集索引（k-center 的初始中心）。
                     AL 第一步若無已標註集，退回隨機選第一個點。
    Returns:
        to_label_data_idx: 選中的未標註索引（長度 num_data_to_label）。
    """
    labeled_idx = labeled_idx or []
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(device)
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=_TF)

    emb_u = _extract_features(feature_extractor, full_dataset, unlabel_data_idx, device,
                              "Coreset: unlabeled feats")          # [Nu, D]
    Nu = emb_u.shape[0]

    # 初始化每個未標註點到「最近已標註中心」的距離
    if labeled_idx:
        emb_l = _extract_features(feature_extractor, full_dataset, labeled_idx, device,
                                  "Coreset: labeled feats")        # [Nl, D]
        min_distances = np.full(Nu, np.inf, dtype=np.float64)
        # 分塊算 unlabeled→labeled 的最近距離，省記憶體
        for s in range(0, len(labeled_idx), 256):
            blk = emb_l[s:s + 256]
            d = np.linalg.norm(emb_u[:, None, :] - blk[None, :, :], axis=2)  # [Nu, blk]
            min_distances = np.minimum(min_distances, d.min(axis=1))
    else:
        # 沒有已標註集（理論上 AL 首輪是 random，不會走到這）→ 隨機第一個點
        min_distances = np.full(Nu, np.inf, dtype=np.float64)
        first = np.random.randint(Nu)
        min_distances = np.minimum(min_distances, np.linalg.norm(emb_u - emb_u[first], axis=1))

    # greedy furthest-first
    selected = []
    for _ in range(num_data_to_label):
        idx = int(np.argmax(min_distances))
        selected.append(idx)
        new_d = np.linalg.norm(emb_u - emb_u[idx], axis=1)
        min_distances = np.minimum(min_distances, new_d)
        min_distances[selected] = -np.inf   # 已選的不再挑

    return [unlabel_data_idx[i] for i in selected]
