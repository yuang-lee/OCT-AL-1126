# =============================================================================
# 附錄：七種主動學習查詢策略（Appendix code snippets）
#
# 設計：採 AL 函式庫慣用的「查詢策略基底類別」模式（如 deep-active-learning /
# baal / ALiPy）。每個方法 = 一個 class，介面完全一致：
#       idx = StrategyName(model, pool, labeled).query(k)
#   - 建構子吃 model、未標註池 pool、已標註集 labeled（後者僅 diversity 類用到）
#   - query(k) 為唯一對外方法，回傳「要標註之點在 pool 中的 index」（長度 k）
#   - 共用的前向推論封裝在基底：proba()→softmax 機率[N,C]、embed(idx)→penultimate 特徵[·,D]
# 三類：Uncertainty(① ② ③)、Diversity(④ ⑤)、Hybrid(⑥ ⑦)。
# =============================================================================
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


# --- 共用基元 ----------------------------------------------------------------
# 前兩者為前向推論（資料載入細節與正文一致，故省略內文）；後三者為純數值工具。
def predict_proba(model, idx):     ...   # 對 idx 推論 → softmax 機率 [len(idx), C]
def embed(model, idx):             ...   # 取 penultimate 特徵       [len(idx), D]
def onehot(y, C):        return np.eye(C)[y]
def l2_normalize(X):     return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
def pairwise_dist(A, B):                 # 歐氏距離矩陣 [len(A), len(B)]
    B = np.atleast_2d(B)
    return np.sqrt(np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T, 0))


class QueryStrategy:
    """所有策略的共同基底。子類別只需覆寫 query()。"""
    def __init__(self, model, pool, labeled=None):
        self.model = model
        self.pool = pool                  # 未標註點的索引
        self.labeled = labeled or []      # 已標註點的索引（diversity 類的初始中心）

    def proba(self):                      # [N, C] softmax 後類別機率（推論細節省略）
        return predict_proba(self.model, self.pool)

    def embed(self, idx):                 # [len(idx), D] penultimate 特徵（ResNet-18：D=512）
        return embed(self.model, idx)

    def query(self, k):                   # → 選中的 k 個 pool index
        raise NotImplementedError


# ======================= Uncertainty =========================================

class LeastConfidence(QueryStrategy):
    """挑「最高類別信心最低」的點。"""
    def query(self, k):
        u = 1.0 - self.proba().max(axis=1)             # 最高信心越低 → 越不確定
        return np.argsort(-u)[:k]


class Entropy(QueryStrategy):
    """挑「預測分布熵最大」的點。"""
    def query(self, k):
        P = self.proba()
        H = -(P * np.log(P + 1e-10)).sum(axis=1)       # H(p) = -Σ p·log p
        return np.argsort(-H)[:k]


class Margin(QueryStrategy):
    """挑「前兩高機率差距最小」的點（最接近決策邊界）。"""
    def query(self, k):
        P = np.sort(self.proba(), axis=1)
        m = P[:, -1] - P[:, -2]                         # top1 - top2；越小越不確定
        return np.argsort(m)[:k]


# ======================= Diversity ===========================================

class Coreset(QueryStrategy):
    """k-Center Greedy (Sener & Savarese, ICLR 2018)。
    特徵空間 furthest-first：每輪挑「離最近已標註中心最遠」者，最大化覆蓋。"""
    def query(self, k):
        Zu = self.embed(self.pool)                      # [Nu, D] 未標註特徵
        Zl = self.embed(self.labeled)                   # [Nl, D] 已標註特徵 = 初始中心
        min_dist = pairwise_dist(Zu, Zl).min(axis=1)    # 每點到最近中心的距離

        selected = []
        for _ in range(k):
            i = int(min_dist.argmax())                  # 離已覆蓋區域最遠者
            selected.append(i)
            min_dist = np.minimum(min_dist, pairwise_dist(Zu, Zu[i]))  # 納入後更新覆蓋半徑
            min_dist[selected] = -np.inf                # 已選的不再挑
        return selected


class TypiClust(QueryStrategy):
    """Hacohen et al., ICML 2022。低預算 diversity：分群後在「未被標註覆蓋」的群裡挑最典型(最稠密)者。"""
    K_NN, MIN_SIZE, MAX_CLUSTERS = 20, 5, 500

    def _typicality(self, Z):                           # [m, D] → 每點密度
        nn = NearestNeighbors(n_neighbors=self.K_NN + 1).fit(Z)
        d, _ = nn.kneighbors(Z)                         # 含自己，故 +1
        return 1.0 / (d[:, 1:].mean(axis=1) + 1e-5)     # 到 K_NN 鄰居越近 → 越典型

    def query(self, k):
        Zu, Zl = self.embed(self.pool), self.embed(self.labeled)
        Z, Nu = np.vstack([Zu, Zl]), len(Zu)            # 前 Nu 個位置是未標註
        n_clusters = min(len(self.labeled) + k, self.MAX_CLUSTERS)
        cl = KMeans(n_clusters).fit_predict(Z)

        # 群排序：已標註數少者優先(挑未覆蓋區)，再大群優先；丟掉 size < MIN_SIZE 的小群
        size  = {c: (cl == c).sum() for c in np.unique(cl)}
        n_lab = {c: ((cl == c) & (np.arange(len(Z)) >= Nu)).sum() for c in size}
        order = sorted([c for c in size if size[c] >= self.MIN_SIZE],
                       key=lambda c: (n_lab[c], -size[c]))

        selected, i = [], 0
        while len(selected) < k:                         # round-robin 走訪各群
            c = order[i % len(order)]; i += 1
            cand = [p for p in np.where(cl == c)[0] if p < Nu and p not in selected]
            if cand:
                selected.append(cand[int(self._typicality(Z[cand]).argmax())])  # 群內最典型的未標註點
        return selected


# ======================= Hybrid ==============================================

class BADGE(QueryStrategy):
    """Ash et al., 2020。最後一層 loss 梯度嵌入 g_x = (onehot(ŷ) - p) ⊗ z，
    再對 {g_x} 做 k-means++：兼顧高梯度範數(不確定)與彼此分散(多樣)。"""
    def query(self, k):
        P, Z = self.proba(), self.embed(self.pool)
        R = onehot(P.argmax(axis=1), P.shape[1]) - P    # label residual；g = R ⊗ z

        # 因式分解免攤平 C·D 維：‖g_i‖²=‖R_i‖²·‖z_i‖²，<g_i,g_j>=<R_i,R_j>·<z_i,z_j>
        g2 = (R ** 2).sum(1) * (Z ** 2).sum(1)          # 各點梯度範數平方
        grad_d2 = lambda c: g2 + g2[c] - 2 * (R @ R[c]) * (Z @ Z[c])   # 到中心 c 的梯度距離²

        centers = [int(g2.argmax())]                    # 第一個中心：梯度範數最大者
        D2 = np.maximum(grad_d2(centers[0]), 0)
        while len(centers) < k:
            D2[centers] = 0
            i = np.random.choice(len(D2), p=D2 / D2.sum())   # D²-weighted 抽樣(k-means++)
            centers.append(int(i))
            D2 = np.minimum(D2, np.maximum(grad_d2(i), 0))
        return centers


class ClusterMargin(QueryStrategy):
    """Citovsky et al., NeurIPS 2021。輕量 hybrid：margin 圈出不確定候選 → HAC 分群 → 跨群取樣去冗餘。"""
    KM_FACTOR, EPS_FRAC = 10, 0.5

    def query(self, k):
        P = np.sort(self.proba(), axis=1)
        cand = np.argsort(P[:, -1] - P[:, -2])[: self.KM_FACTOR * k]   # ① margin 最小的 k_m=10k 候選
        Z = l2_normalize(self.embed(self.pool)[cand])

        # ② 對候選做 HAC(average linkage)；ε = 候選兩兩距離中位數 × EPS_FRAC（尺度自適應）
        eps = np.median(pairwise_dist(Z, Z)[np.triu_indices(len(Z), 1)]) * self.EPS_FRAC
        clusters = AgglomerativeClustering(n_clusters=None, linkage="average",
                                           distance_threshold=eps).fit_predict(Z)

        # ③ 群按大小升冪 round-robin、每群隨機取一，湊滿 k（小群先取，兼顧不確定與多樣）
        groups = {c: list(np.where(clusters == c)[0]) for c in np.unique(clusters)}
        for g in groups.values():
            np.random.shuffle(g)
        order = sorted(groups, key=lambda c: len(groups[c]))

        selected = []
        while len(selected) < k:
            for c in order:
                if groups[c]:
                    selected.append(int(groups[c].pop()))
                    if len(selected) == k:
                        break
        return cand[selected]


# =============================================================================
# 統一呼叫（七種介面相同，可直接以名稱 dispatch）：
#   STRATEGIES = {"least_conf": LeastConfidence, "entropy": Entropy, "margin": Margin,
#                 "coreset": Coreset, "typiclust": TypiClust,
#                 "badge": BADGE, "cluster_margin": ClusterMargin}
#   idx = STRATEGIES[name](model, pool, labeled).query(k)   # → pool 中要標註的 k 個 index
# =============================================================================
