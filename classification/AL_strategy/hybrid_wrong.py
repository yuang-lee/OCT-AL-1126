import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

def distance_vectorized(X1, X2, mu):
    """
    向量化的距离计算，提高计算效率
    """
    candidate_probs, candidate_prob_norm = X1
    candidate_emb, candidate_emb_norm = X2
    (center_probs, center_prob_norm_sq), (center_emb, center_emb_norm_sq) = mu
    
    # 使用 numpy 的向量化操作
    dot_prob = candidate_probs @ center_probs
    dot_emb = candidate_emb @ center_emb
    
    # 向量化距离计算
    dist_sq = (candidate_prob_norm * center_prob_norm_sq + 
               candidate_emb_norm * center_emb_norm_sq - 
               2 * dot_prob * dot_emb)
    
    dist_sq = np.maximum(dist_sq, 0)  # 使用 maximum 替代 clip
    return np.sqrt(dist_sq)

def init_centers_optimized(X1, X2, chosen, chosen_list, mu, D2):
    """
    优化的中心初始化函数
    """
    if len(chosen) == 0:
        # 初始中心选择
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance_vectorized(X1, X2, mu[0]).astype(np.float32)  # 使用 float32 节省内存
        D2[ind] = 0
    else:
        # 计算到新中心的距离
        newD = distance_vectorized(X1, X2, mu[-1]).astype(np.float32)
        D2 = np.minimum(D2, newD)
        
        # 将已选择的点的距离设为 0
        D2[chosen_list] = 0  # 向量化操作
        
        # 优化的概率采样
        D2_squared = D2 ** 2
        prob_sum = np.sum(D2_squared)
        
        if prob_sum == 0:
            # 如果所有距离都为0，随机选择
            available_indices = np.setdiff1d(np.arange(len(D2)), chosen_list)
            ind = np.random.choice(available_indices)
        else:
            # 使用更高效的采样方法
            probs = D2_squared / prob_sum
            ind = np.random.choice(len(probs), p=probs)
            
            # 确保不重复选择
            max_attempts = 100
            attempts = 0
            while ind in chosen and attempts < max_attempts:
                ind = np.random.choice(len(probs), p=probs)
                attempts += 1
            
            if attempts >= max_attempts:
                # 如果重复选择太多次，直接从未选择的点中随机选一个
                available_indices = np.setdiff1d(np.arange(len(D2)), chosen_list)
                ind = np.random.choice(available_indices)
        
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    
    chosen.add(ind)
    chosen_list.append(ind)
    return chosen, chosen_list, mu, D2

def badge_optimized(model, data_dir, unlabel_data_idx, num_data_to_label, device, batch_size=64):
    """
    优化的 BADGE 采样函数
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)

    model.eval()
    model.to(device)

    # 创建特征提取器
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()
    feature_extractor.to(device)

    all_embeddings = []
    all_probs = []
    
    # 使用更大的 batch size 和 pin_memory 加速数据加载
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="extracting model embeddings and logits..."):
            inputs = inputs.to(device, non_blocking=True)
            
            # 获取嵌入
            embeddings = feature_extractor(inputs)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.view(embeddings.size(0), -1)
            
            # 获取概率
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_probs.append(probs.cpu())
    
    # 转换为 numpy 数组，使用 float32 节省内存
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    all_probs = torch.cat(all_probs, dim=0).numpy().astype(np.float32)
    N = all_embeddings.shape[0]

    # 预计算范数
    emb_norm_square = np.sum(all_embeddings ** 2, axis=1)
    prob_norm_square = np.sum(all_probs ** 2, axis=1)

    # 计算修改后的概率
    max_inds = np.argmax(all_probs, axis=1)
    modified_probs = -all_probs.copy()
    modified_probs[np.arange(N), max_inds] += 1

    X1 = (modified_probs, prob_norm_square)
    X2 = (all_embeddings, emb_norm_square)

    chosen = set()
    chosen_list = []
    mu = None
    D2 = None

    # K-means++ 初始化
    print(f"Starting k-means++ initialization for {num_data_to_label} centers...")
    for i in tqdm(range(num_data_to_label), desc='K-means++ center selection'):
        chosen, chosen_list, mu, D2 = init_centers_optimized(X1, X2, chosen, chosen_list, mu, D2)
        
        # 每10个中心输出一次进度
        if (i + 1) % 10 == 0:
            print(f"Selected {i + 1}/{num_data_to_label} centers")

    to_label_data_idx = [unlabel_data_idx[i] for i in chosen_list]
    return to_label_data_idx

# 保持原来的函数名，直接使用优化版本
def badge(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    """
    原始的 badge 函数名，但使用优化的实现
    """
    return badge_optimized(model, data_dir, unlabel_data_idx, num_data_to_label, device)

# 如果需要进一步优化，可以考虑以下方案：
def badge_with_approximation(model, data_dir, unlabel_data_idx, num_data_to_label, device, 
                           subsample_ratio=0.1):
    """
    使用子采样近似的 BADGE 方法，适用于超大数据集
    """
    # 如果数据集太大，先进行随机子采样
    if len(unlabel_data_idx) > 10000:
        subsample_size = max(int(len(unlabel_data_idx) * subsample_ratio), num_data_to_label * 10)
        subsample_indices = np.random.choice(len(unlabel_data_idx), subsample_size, replace=False)
        subsample_unlabel_idx = [unlabel_data_idx[i] for i in subsample_indices]
        
        # 在子样本上运行 BADGE
        selected_subsample_idx = badge_optimized(model, data_dir, subsample_unlabel_idx, 
                                                num_data_to_label, device)
        return selected_subsample_idx
    else:
        return badge_optimized(model, data_dir, unlabel_data_idx, num_data_to_label, device)