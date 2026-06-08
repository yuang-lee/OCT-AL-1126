import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms



def compute_density_scores(target_idx, feat_dict, k: int = 10, device: str = "cpu"):
    feats = torch.stack([torch.from_numpy(feat_dict[idx]) for idx in target_idx])  # shape (N, D)
    feats = feats.to(device)

    id2row = {idx: i for i, idx in enumerate(target_idx)}

    density = {}
    k = max(1, min(k, feats.shape[0] - 1))

    for sample_id in tqdm(target_idx, desc="Computing density scores"):
        row = id2row[sample_id]
        sample_feat = feats[row : row + 1]  # shape (1, D)

        dists = torch.cdist(sample_feat, feats, p=2).squeeze(0)  # shape (N,)
        dists[row] = float("inf")  # exclude self-distance

        knn_distances, _ = torch.topk(dists, k, largest=False)
        avg_dist = knn_distances.mean().item()
        density_score = 1.0 / (avg_dist + 1e-8)  # avoid divide-by-zero
        density[sample_id] = float(density_score)

    return density


def coreset(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    """
    基於官方實現優化的 K-Center Greedy 演算法進行 active learning 資料選擇
    
    主要優化：
    1. 預計算嵌入向量並緩存
    2. 使用矩陣乘法優化距離計算
    3. 增量式更新最小距離
    4. 提高批次處理效率
    
    Args:
        model: 訓練好的模型
        data_dir: 資料目錄路徑
        unlabel_data_idx: 未標記資料的索引列表
        num_data_to_label: 要選擇進行標記的資料數量
        device: 計算設備 (cuda/cpu)
    
    Returns:
        to_label_data_idx: 選中要標記的資料索引列表
    """
    print(f"開始 Coreset 選擇，從 {len(unlabel_data_idx)} 個樣本中選擇 {num_data_to_label} 個")
    
    # ========== Stage 1: 特徵提取優化 ==========
    # 建立特徵提取器 (移除最後的分類層)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    # 優化資料載入參數
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    
    # 動態調整批次大小和工作進程數
    optimal_batch_size = min(128, len(unlabel_dataset), 2 ** int(np.log2(len(unlabel_dataset) / 10)))
    optimal_num_workers = min(8, torch.get_num_threads())
    
    unlabel_loader = DataLoader(
        unlabel_dataset, 
        batch_size=optimal_batch_size, 
        shuffle=False, 
        num_workers=optimal_num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if optimal_num_workers > 0 else 2
    )

    # 提取特徵 (預計算並緩存)
    print("提取特徵嵌入向量...")
    start_time = time.time()
    
    features = []
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="Extracting embeddings"):
            inputs = inputs.to(device, non_blocking=True)
            feats = feature_extractor(inputs)
            feats = feats.view(feats.size(0), -1)  # flatten
            features.append(feats.cpu().numpy())  # 直接轉換為 numpy

    # 合併所有特徵，避免多次 torch.cat
    embeddings = np.vstack(features)  # shape: [N_unlabeled, feature_dim]
    
    print(f"特徵提取完成，耗時: {time.time() - start_time:.2f}秒")
    print(f"特徵維度: {embeddings.shape}")
    
    # ========== Stage 2: K-Center Greedy 演算法優化 ==========
    print("執行優化版 K-Center Greedy 演算法...")
    start_time = time.time()
    
    n_samples = len(unlabel_data_idx)
    
    # 第一個點隨機選擇
    selected_indices = []
    first_idx = np.random.randint(n_samples)
    selected_indices.append(first_idx)
    
    if num_data_to_label == 1:
        return [unlabel_data_idx[first_idx]]
    
    # 選擇最適合的距離計算方法
    if n_samples <= 5000:  # 小資料集：預計算完整距離矩陣
        print("使用預計算距離矩陣方法...")
        dist_matrix = _compute_fast_euclidean_distance_matrix(embeddings)
        selected_indices = _kcenter_greedy_with_precomputed_distances(
            dist_matrix, num_data_to_label, first_idx
        )
    else:  # 大資料集：增量式距離計算
        print("使用增量式距離計算方法...")
        selected_indices = _kcenter_greedy_incremental(
            embeddings, num_data_to_label, first_idx
        )
    
    print(f"K-Center 演算法完成，耗時: {time.time() - start_time:.2f}秒")
    
    # 對應回原始索引
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    print(f"選擇完成！選中的前5個樣本索引: {to_label_data_idx[:5]}")
    return to_label_data_idx


def _compute_fast_euclidean_distance_matrix(embeddings):
    """
    使用矩陣乘法優化的歐幾里得距離計算
    基於官方實現：||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    """
    # 計算內積矩陣
    dot_product = np.matmul(embeddings, embeddings.T)
    
    # 計算平方範數
    square_norm = np.array(np.diag(dot_product)).reshape(-1, 1)
    
    # 使用廣播計算平方距離
    squared_distances = square_norm + square_norm.T - 2 * dot_product
    
    # 確保數值穩定性（避免負數）
    squared_distances = np.maximum(squared_distances, 0)
    
    # 計算歐幾里得距離
    distances = np.sqrt(squared_distances)
    
    return distances


def _kcenter_greedy_with_precomputed_distances(dist_matrix, num_data_to_label, first_idx):
    """
    使用預計算距離矩陣的 K-Center Greedy 演算法
    """
    n_samples = dist_matrix.shape[0]
    selected = [first_idx]
    
    # 初始化最小距離（到第一個選中點的距離）
    min_distances = dist_matrix[:, first_idx].copy()
    
    # 貪心選擇剩餘點
    for _ in range(num_data_to_label - 1):
        # 選擇距離已選點最遠的點
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
        
        # 更新最小距離（增量式更新）
        new_distances = dist_matrix[:, next_idx]
        min_distances = np.minimum(min_distances, new_distances)
    
    return selected


def _kcenter_greedy_incremental(embeddings, num_data_to_label, first_idx):
    """
    增量式 K-Center Greedy 演算法，適用於大資料集
    """
    n_samples = embeddings.shape[0]
    selected = [first_idx]
    
    # 計算到第一個點的距離
    first_point = embeddings[first_idx:first_idx+1]  # shape: (1, feature_dim)
    min_distances = np.linalg.norm(embeddings - first_point, axis=1)
    
    # 貪心選擇剩餘點
    for _ in range(num_data_to_label - 1):
        # 選擇距離已選點最遠的點
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
        
        # 計算新選點到所有點的距離
        new_point = embeddings[next_idx:next_idx+1]  # shape: (1, feature_dim)
        new_distances = np.linalg.norm(embeddings - new_point, axis=1)
        
        # 更新最小距離
        min_distances = np.minimum(min_distances, new_distances)
    
    return selected


# def _compute_distances_vectorized_batch(embeddings, selected_points, batch_size=1000):
#     """
#     分批計算距離，適用於超大資料集，避免記憶體溢出
#     """
#     n_samples = embeddings.shape[0]
#     min_distances = np.full(n_samples, np.inf)
    
#     for i in range(0, n_samples, batch_size):
#         end_idx = min(i + batch_size, n_samples)
#         batch_embeddings = embeddings[i:end_idx]
        
#         # 計算這個批次到所有已選點的距離
#         if len(selected_points.shape) == 1:  # 單個點
#             distances = np.linalg.norm(
#                 batch_embeddings - selected_points.reshape(1, -1), axis=1
#             )
#         else:  # 多個點
#             distances = np.linalg.norm(
#                 batch_embeddings[:, None, :] - selected_points[None, :, :], axis=2
#             )
#             distances = np.min(distances, axis=1)
        
#         min_distances[i:end_idx] = np.minimum(min_distances[i:end_idx], distances)
    
#     return min_distances


# # 超大資料集版本（記憶體優化）
# def coreset_memory_efficient(model, data_dir, unlabel_data_idx, num_data_to_label, device, max_memory_gb=8):
#     """
#     記憶體優化版本，適用於超大資料集
#     """
#     # 估算可用記憶體
#     feature_size = 512  # 假設特徵維度
#     max_samples_in_memory = int((max_memory_gb * 1024**3) / (feature_size * 8))  # 8 bytes per float64
    
#     if len(unlabel_data_idx) <= max_samples_in_memory:
#         # 記憶體足夠，使用標準版本
#         return coreset_optimized_official(model, data_dir, unlabel_data_idx, num_data_to_label, device)
    
#     else:
#         # 記憶體不足，使用分批處理
#         print(f"資料集過大 ({len(unlabel_data_idx)} 樣本)，使用記憶體優化版本")
        
#         # 分批提取特徵並選擇
#         batch_size = max_samples_in_memory // 2
#         selected_global = []
        
#         for i in range(0, len(unlabel_data_idx), batch_size):
#             end_idx = min(i + batch_size, len(unlabel_data_idx))
#             batch_indices = unlabel_data_idx[i:end_idx]
            
#             # 在這個批次中選擇
#             batch_selected = coreset_optimized_official(
#                 model, data_dir, batch_indices, 
#                 min(num_data_to_label // 4, len(batch_indices)), device
#             )
#             selected_global.extend(batch_selected)
        
#         # 從預選的點中再次選擇
#         if len(selected_global) > num_data_to_label:
#             final_selected = coreset_optimized_official(
#                 model, data_dir, selected_global, num_data_to_label, device
#             )
#             return final_selected
        
#         return selected_global[:num_data_to_label]





# import numpy as np
# import torch
# from tqdm import tqdm
# from torch.utils.data import Subset, DataLoader
# from torchvision import datasets, transforms
# from sklearn.metrics import pairwise_distances
# from tqdm import tqdm



# def coreset(model, data_dir, unlabel_data_idx, num_data_to_label, device):
#     """
#     使用 K-Center Greedy 演算法進行 active learning 資料選擇
    
#     Args:
#         model: 訓練好的模型
#         data_dir: 資料目錄路徑
#         unlabel_data_idx: 未標記資料的索引列表
#         num_data_to_label: 要選擇進行標記的資料數量
#         device: 計算設備 (cuda/cpu)
    
#     Returns:
#         to_label_data_idx: 選中要標記的資料索引列表
#     """
#     # 建立特徵提取器 (移除最後的分類層)
#     feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#     feature_extractor.eval()
#     feature_extractor.to(device)

#     # 資料預處理
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
    
#     # 載入資料
#     full_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
#     unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
#     unlabel_loader = DataLoader(unlabel_dataset, batch_size=32, shuffle=False, num_workers=4)

#     # Step 1: 提取特徵嵌入向量
#     features = []
#     for inputs, _ in tqdm(unlabel_loader, desc="Extracting embeddings for CORESET"):
#         inputs = inputs.to(device)
#         with torch.no_grad():
#             feats = feature_extractor(inputs)   # output shape: (B, 512, 1, 1)
#             feats = feats.view(feats.size(0), -1)  # flatten to (B, 512)
#         features.append(feats.cpu())

#     features = torch.cat(features, dim=0).numpy()  # shape: [N_unlabeled, 512]

#     # Step 2: K-Center Greedy 演算法 (改進版)
#     selected = []
#     num_unlabeled = len(unlabel_data_idx)
    
#     # 第一個點隨機選擇
#     first_idx = np.random.randint(num_unlabeled)
#     selected.append(first_idx)
    
#     # 如果只需要選一個點，直接返回
#     if num_data_to_label == 1:
#         to_label_data_idx = [unlabel_data_idx[first_idx]]
#         return to_label_data_idx
    
#     # 初始化每個點到已選點的最小距離
#     min_distances = pairwise_distances(
#         features, features[first_idx:first_idx+1], metric='euclidean'
#     ).flatten()
    
#     # 貪心選擇剩餘的點
#     for _ in range(num_data_to_label - 1):
#         # 選擇距離已選點最遠的點
#         next_idx = np.argmax(min_distances)
#         selected.append(next_idx)
        
#         # 計算新選點到所有點的距離
#         new_dists = pairwise_distances(
#             features, features[next_idx:next_idx+1], metric='euclidean'
#         ).flatten()
        
#         # 更新每個點到已選點集合的最小距離
#         min_distances = np.minimum(min_distances, new_dists)

#     # 將選中的索引對應回原始訓練集中的索引
#     to_label_data_idx = [unlabel_data_idx[i] for i in selected]
#     return to_label_data_idx