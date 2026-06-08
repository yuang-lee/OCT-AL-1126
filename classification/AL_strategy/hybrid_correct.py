"""
BADGE (Ash et al. 2020, arXiv:1906.03671) — 正確版。

梯度嵌入 g_x = (p - onehot(argmax)) ⊗ feature（penultimate）。
本實作不顯式攤平 C·D 維，而用因式分解：
  <g_i, g_j> = <mp_i, mp_j> · <emb_i, emb_j>
  ‖g_i‖²   = <g_i, g_i> = ‖mp_i‖² · ‖emb_i‖²        （同一點的兩個範數相乘）
故 ‖g_i - g_j‖² = ‖mp_i‖²·‖emb_i‖² + ‖mp_j‖²·‖emb_j‖² - 2·<mp_i,mp_j>·<emb_i,emb_j>。

對 hybrid_wrong.py 的修正：原本範數項寫成
  ‖mp_i‖²·‖mp_center‖² + ‖emb_i‖²·‖emb_center‖²（把 i 與 center 的範數交叉相乘）
是錯的（自身距離 ≠ 0）。此處改為 ‖mp_i‖²·‖emb_i‖² + ‖mp_c‖²·‖emb_c‖²。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


def distance_vectorized(X1, X2, mu):
    """到單一 center 的梯度嵌入歐氏距離（正確公式）。"""
    candidate_probs, candidate_prob_norm = X1          # mp_i [N,C], ‖mp_i‖² [N]
    candidate_emb, candidate_emb_norm = X2             # emb_i [N,D], ‖emb_i‖² [N]
    (center_probs, center_prob_norm_sq), (center_emb, center_emb_norm_sq) = mu
    dot_prob = candidate_probs @ center_probs          # <mp_i, mp_c> [N]
    dot_emb = candidate_emb @ center_emb               # <emb_i, emb_c> [N]
    dist_sq = (candidate_prob_norm * candidate_emb_norm        # ‖g_i‖²
               + center_prob_norm_sq * center_emb_norm_sq      # ‖g_c‖²
               - 2 * dot_prob * dot_emb)                       # 2<g_i,g_c>
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)


def init_centers_optimized(X1, X2, chosen, chosen_list, mu, D2):
    """k-means++ seeding（與 BADGE 一致）。"""
    if len(chosen) == 0:
        # 第一個 center：取梯度嵌入範數最大者 ‖g‖² = ‖mp‖²·‖emb‖²
        ind = int(np.argmax(X1[1] * X2[1]))
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance_vectorized(X1, X2, mu[0]).astype(np.float32)
        D2[ind] = 0
    else:
        newD = distance_vectorized(X1, X2, mu[-1]).astype(np.float32)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        D2_squared = D2 ** 2
        prob_sum = float(np.sum(D2_squared))
        if prob_sum == 0:
            available = np.setdiff1d(np.arange(len(D2)), chosen_list)
            ind = int(np.random.choice(available))
        else:
            probs = D2_squared / prob_sum
            ind = int(np.random.choice(len(probs), p=probs))
            attempts = 0
            while ind in chosen and attempts < 100:
                ind = int(np.random.choice(len(probs), p=probs))
                attempts += 1
            if attempts >= 100:
                available = np.setdiff1d(np.arange(len(D2)), chosen_list)
                ind = int(np.random.choice(available))
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    return chosen, chosen_list, mu, D2


def badge(model, data_dir, unlabel_data_idx, num_data_to_label, device, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

    model.eval().to(device)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False

    all_embeddings, all_probs = [], []
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="BADGE: embeddings + logits"):
            inputs = inputs.to(device, non_blocking=True)
            emb = feature_extractor(inputs)
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            probs = F.softmax(model(inputs), dim=1)
            all_embeddings.append(emb.cpu())
            all_probs.append(probs.cpu())
    all_embeddings = torch.cat(all_embeddings).numpy().astype(np.float32)
    all_probs = torch.cat(all_probs).numpy().astype(np.float32)

    N = all_embeddings.shape[0]
    emb_norm_square = np.sum(all_embeddings ** 2, axis=1)
    # 梯度的 label-residual 部分：mp = onehot(argmax) - p
    max_inds = np.argmax(all_probs, axis=1)
    mp = -all_probs.copy()
    mp[np.arange(N), max_inds] += 1
    prob_norm_square = np.sum(mp ** 2, axis=1)

    X1 = (mp, prob_norm_square)
    X2 = (all_embeddings, emb_norm_square)
    chosen, chosen_list, mu, D2 = set(), [], None, None
    for _ in tqdm(range(num_data_to_label), desc="BADGE: k-means++"):
        chosen, chosen_list, mu, D2 = init_centers_optimized(X1, X2, chosen, chosen_list, mu, D2)

    return [unlabel_data_idx[i] for i in chosen_list]
