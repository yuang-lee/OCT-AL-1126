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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


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
    # ResNetSimCLR 的 children 只有 .backbone（resnet18，fc 已換成分類頭）；要進 backbone 拿
    # penultimate 512 維特徵，否則 Sequential(children[:-1]) 變 identity → 吐原始影像(1.8M維) → OOM。
    net = model.backbone if hasattr(model, "backbone") else model
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1]).eval().to(device)
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=_TF)

    emb_u = _extract_features(feature_extractor, full_dataset, unlabel_data_idx, device,
                              "Coreset: unlabeled feats")          # [Nu, D]
    Nu = emb_u.shape[0]

    # 初始化每個未標註點到「最近已標註中心」的距離
    if labeled_idx:
        emb_l = _extract_features(feature_extractor, full_dataset, labeled_idx, device,
                                  "Coreset: labeled feats")        # [Nl, D]
        # ||u-l||² = ||u||² + ||l||² - 2 u·l；用 matmul 得 [Nu,Nl]（小），免 3D broadcast 爆記憶體
        uu = np.sum(emb_u ** 2, axis=1)                            # [Nu]
        ll = np.sum(emb_l ** 2, axis=1)                            # [Nl]
        d2 = uu[:, None] + ll[None, :] - 2.0 * (emb_u @ emb_l.T)   # [Nu, Nl]
        min_distances = np.sqrt(np.maximum(d2.min(axis=1), 0.0)).astype(np.float64)
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


# =============================================================================
# TypiClust (Hacohen, Dekel, Weinshall, ICML 2022) — 低預算 diversity。
# 官方碼：github.com/avihu111/TypiClust。對 (已標註+未標註) 特徵做 KMeans，
# 在「未被標註覆蓋、又大、又密(typical)」的群裡挑代表點。
#   typicality = 1 / (到 K_NN 最近鄰的平均距離 + 1e-5)。
# 與 coreset 互補：coreset 走 geometry(furthest)，typiclust 走 density(typical)。
# =============================================================================
_K_NN = 20
_MIN_CLUSTER_SIZE = 5
_MAX_NUM_CLUSTERS = 500


def _typicality(feats, k):
    """feats: [m, D] → 每點 typicality（1/到 k 最近鄰平均距離）。"""
    m = feats.shape[0]
    if m == 1:
        return np.array([1.0])
    k = max(1, min(k, m - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)   # +1 因為含自己
    dist, _ = nn.kneighbors(feats)
    mean_d = dist[:, 1:].mean(axis=1)                     # 去掉第 0 欄(自己)
    return 1.0 / (mean_d + 1e-5)


def typiclust(model, data_dir, unlabel_data_idx, num_data_to_label, device, labeled_idx=None):
    """
    Args:
        labeled_idx: 目前已標註索引；用來把含已標註的群往後排（優先挑未覆蓋區域）。
    Returns:
        選中的未標註索引（長度 num_data_to_label）。
    """
    labeled_idx = labeled_idx or []
    net = model.backbone if hasattr(model, "backbone") else model
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1]).eval().to(device)
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=_TF)

    emb_u = _extract_features(feature_extractor, full_dataset, unlabel_data_idx, device,
                              "Typiclust: unlabeled feats")        # [Nu, D]
    Nu = emb_u.shape[0]
    if labeled_idx:
        emb_l = _extract_features(feature_extractor, full_dataset, labeled_idx, device,
                                  "Typiclust: labeled feats")      # [Nl, D]
        feats = np.vstack([emb_u, emb_l])
    else:
        feats = emb_u
    is_unlabeled = np.arange(feats.shape[0]) < Nu     # 前 Nu 個位置是未標註

    budget = num_data_to_label
    n_clusters = int(min(len(labeled_idx) + budget, _MAX_NUM_CLUSTERS))
    n_clusters = max(1, min(n_clusters, feats.shape[0]))
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=5000, n_init=3)
    cl = km.fit_predict(feats)

    clusters = {}
    for c in np.unique(cl):
        members = np.where(cl == c)[0]
        clusters[c] = {"members": members, "size": len(members),
                       "n_lab": int(np.sum(~is_unlabeled[members]))}
    # 只保留 size>=MIN 的群（不足則放寬到全部）；排序：已標註少優先、再大群優先
    elig = [c for c in clusters if clusters[c]["size"] >= _MIN_CLUSTER_SIZE]
    if not elig:
        elig = list(clusters.keys())
    order = sorted(elig, key=lambda c: (clusters[c]["n_lab"], -clusters[c]["size"]))

    selected, i, guard = [], 0, 0
    max_guard = budget * 50 + len(order) + 10
    seen = set()
    while len(selected) < budget and guard < max_guard:
        guard += 1
        c = order[i % len(order)]
        i += 1
        cand = [p for p in clusters[c]["members"] if p < Nu and p not in seen]
        if not cand:
            continue
        cand = np.array(cand)
        typ = _typicality(feats[cand], _K_NN)
        pick = int(cand[int(np.argmax(typ))])
        selected.append(pick)
        seen.add(pick)

    return [unlabel_data_idx[p] for p in selected]
