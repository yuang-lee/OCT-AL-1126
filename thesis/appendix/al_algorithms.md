# 附錄　主動學習查詢策略之實作

本附錄列出論文所評估之七種主動學習（active learning）查詢策略的實作。為使各策略易於比較，
本研究將其統一為相同的程式介面：每一種策略以一個類別（class）封裝，繼承自共同的基底類別
`QueryStrategy`，並僅覆寫單一對外方法 `query(k)`。該方法回傳本輪欲標註之樣本於未標註池
中的索引，長度為查詢預算 $k$。此設計沿用主動學習函式庫常見的「查詢策略基底類別」模式，
使七種策略具有一致的呼叫與回傳形式：

```python
idx = StrategyName(model, pool, labeled).query(k)
```

以下程式碼著重於演算法本身；資料載入、前向推論等與策略無關的細節，皆封裝於共用基元中。

---

## A.1　共用基底與基元

所有策略共用兩項前向推論基元與三項數值工具。`predict_proba` 對未標註樣本推論並回傳
softmax 機率，`embed` 取出分類層前一層（penultimate layer）的特徵；於 ResNet-18 中其維度為
512。不確定度類策略僅需前者，多樣性與混合類策略則另需後者所提供的特徵表示。

```python
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


# 前向推論基元（推論細節與正文一致，於此省略）
def predict_proba(model, idx): ...     # 回傳 softmax 機率，形狀 [len(idx), C]
def embed(model, idx):         ...     # 回傳 penultimate 特徵，形狀 [len(idx), D]

# 數值工具
def onehot(y, C):        return np.eye(C)[y]
def l2_normalize(X):     return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
def pairwise_dist(A, B):
    B = np.atleast_2d(B)
    return np.sqrt(np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T, 0))


class QueryStrategy:
    """查詢策略基底。建構子接收模型、未標註池 pool 與已標註集 labeled；
    子類別覆寫 query(k)，回傳欲標註樣本於 pool 中的索引。"""
    def __init__(self, model, pool, labeled=None):
        self.model = model
        self.pool = pool
        self.labeled = labeled or []

    def proba(self):          return predict_proba(self.model, self.pool)
    def embed(self, idx):     return embed(self.model, idx)

    def query(self, k):
        raise NotImplementedError
```

---

## A.2　不確定度策略（Uncertainty）

不確定度策略僅依模型的預測機率排序，挑選模型最為猶豫的樣本，計算成本低廉。

### A.2.1　最小信心度（Least Confidence）

選取最高類別機率最低之樣本，即模型對其最有把握的類別仍信心不足者。

```python
class LeastConfidence(QueryStrategy):
    def query(self, k):
        u = 1.0 - self.proba().max(axis=1)             # 最高信心越低，不確定度越高
        return np.argsort(-u)[:k]
```

### A.2.2　最大熵（Entropy）

以預測分布之夏農熵 $H(p) = -\sum_i p_i \log p_i$ 為不確定度，選取熵最大者；相較最小信心度，
熵同時考量整個機率分布的離散程度。

```python
class Entropy(QueryStrategy):
    def query(self, k):
        P = self.proba()
        H = -(P * np.log(P + 1e-10)).sum(axis=1)
        return np.argsort(-H)[:k]
```

### A.2.3　邊際採樣（Margin）

以最高與次高類別機率之差 $p_{(1)} - p_{(2)}$ 為不確定度指標，差距越小代表樣本越接近決策邊界。

```python
class Margin(QueryStrategy):
    def query(self, k):
        P = np.sort(self.proba(), axis=1)
        m = P[:, -1] - P[:, -2]                        # top1 與 top2 之差，越小越不確定
        return np.argsort(m)[:k]
```

---

## A.3　多樣性策略（Diversity）

多樣性策略於特徵空間中選樣，著重所選樣本對資料分布的覆蓋，而不直接依賴模型的預測信心。

### A.3.1　Coreset（k-Center Greedy）

依 Sener 與 Savarese（ICLR 2018）所提之 k-center 貪婪法，於特徵空間進行最遠點優先選取。
以現有已標註樣本為初始中心，每一輪選取「距離最近中心最遠」之未標註點並更新各點之最近中心距離，
逐步最小化未標註集對已選集合的最大距離。

```python
class Coreset(QueryStrategy):
    def query(self, k):
        Zu = self.embed(self.pool)                     # 未標註特徵
        Zl = self.embed(self.labeled)                  # 已標註特徵，作為初始中心
        min_dist = pairwise_dist(Zu, Zl).min(axis=1)   # 各點至最近中心之距離

        selected = []
        for _ in range(k):
            i = int(min_dist.argmax())                 # 最遠點優先
            selected.append(i)
            min_dist = np.minimum(min_dist, pairwise_dist(Zu, Zu[i]))
            min_dist[selected] = -np.inf
        return selected
```

### A.3.2　TypiClust

依 Hacohen 等人（ICML 2022）所提之低預算多樣性策略。對已標註與未標註樣本之特徵共同分群後，
優先選擇尚未被標註涵蓋之群集，並於各群集中挑選最具代表性（最典型）之樣本。樣本之典型度定義為
其與 $K$ 個最近鄰平均距離之倒數，距離越小者所處區域越稠密、越具代表性。

```python
class TypiClust(QueryStrategy):
    K_NN, MIN_SIZE, MAX_CLUSTERS = 20, 5, 500

    def _typicality(self, Z):
        nn = NearestNeighbors(n_neighbors=self.K_NN + 1).fit(Z)
        d, _ = nn.kneighbors(Z)                        # 第 0 欄為樣本自身，故取 +1 鄰居
        return 1.0 / (d[:, 1:].mean(axis=1) + 1e-5)

    def query(self, k):
        Zu, Zl = self.embed(self.pool), self.embed(self.labeled)
        Z, Nu = np.vstack([Zu, Zl]), len(Zu)           # 前 Nu 列為未標註樣本
        n_clusters = min(len(self.labeled) + k, self.MAX_CLUSTERS)
        cl = KMeans(n_clusters).fit_predict(Z)

        # 群集排序：已標註樣本數較少者優先，其次為較大群集；過小群集予以排除
        size  = {c: (cl == c).sum() for c in np.unique(cl)}
        n_lab = {c: ((cl == c) & (np.arange(len(Z)) >= Nu)).sum() for c in size}
        order = sorted([c for c in size if size[c] >= self.MIN_SIZE],
                       key=lambda c: (n_lab[c], -size[c]))

        selected, i = [], 0
        while len(selected) < k:                       # 逐群輪流選取
            c = order[i % len(order)]; i += 1
            cand = [p for p in np.where(cl == c)[0] if p < Nu and p not in selected]
            if cand:
                selected.append(cand[int(self._typicality(Z[cand]).argmax())])
        return selected
```

---

## A.4　混合策略（Hybrid）

混合策略同時考量不確定度與多樣性，期能避免不確定度策略選取大量相似困難樣本所造成的冗餘。

### A.4.1　BADGE

依 Ash 等人（2020）所提之方法。以分類層對交叉熵損失之梯度作為樣本嵌入，
$g_x = (\,\text{onehot}(\hat{y}) - p\,) \otimes z$，其範數反映預測不確定度、方向反映樣本特性。
其後對梯度嵌入施以 k-means++ 種子選取，使所選樣本兼具高梯度範數與彼此分散之特性。實作上利用
$\lVert g\rVert^2 = \lVert R\rVert^2\lVert z\rVert^2$ 之因式分解，避免顯式展開 $C\times D$ 維向量。

```python
class BADGE(QueryStrategy):
    def query(self, k):
        P, Z = self.proba(), self.embed(self.pool)
        R = onehot(P.argmax(axis=1), P.shape[1]) - P   # 損失對 logits 之殘差項

        g2 = (R ** 2).sum(1) * (Z ** 2).sum(1)         # 梯度嵌入之範數平方
        grad_d2 = lambda c: g2 + g2[c] - 2 * (R @ R[c]) * (Z @ Z[c])

        centers = [int(g2.argmax())]                   # 首一中心取梯度範數最大者
        D2 = np.maximum(grad_d2(centers[0]), 0)
        while len(centers) < k:
            D2[centers] = 0
            i = np.random.choice(len(D2), p=D2 / D2.sum())   # 依距離平方加權抽樣
            centers.append(int(i))
            D2 = np.minimum(D2, np.maximum(grad_d2(i), 0))
        return centers
```

### A.4.2　Cluster-Margin

依 Citovsky 等人（NeurIPS 2021）所提之方法，為計算上較為輕量之混合策略。首先依邊際採樣取出
最不確定之 $k_m = 10k$ 個候選樣本，繼以階層式聚合分群（average linkage）將候選分群，最後依群集
大小由小至大輪流自各群取樣，以去除候選間之冗餘。分群門檻 $\varepsilon$ 取候選兩兩距離之中位數
乘以一比例係數，使其隨特徵尺度自適應。

```python
class ClusterMargin(QueryStrategy):
    KM_FACTOR, EPS_FRAC = 10, 0.5

    def query(self, k):
        P = np.sort(self.proba(), axis=1)
        cand = np.argsort(P[:, -1] - P[:, -2])[: self.KM_FACTOR * k]   # 取邊際最小之候選
        Z = l2_normalize(self.embed(self.pool)[cand])

        eps = np.median(pairwise_dist(Z, Z)[np.triu_indices(len(Z), 1)]) * self.EPS_FRAC
        clusters = AgglomerativeClustering(n_clusters=None, linkage="average",
                                           distance_threshold=eps).fit_predict(Z)

        groups = {c: list(np.where(clusters == c)[0]) for c in np.unique(clusters)}
        for g in groups.values():
            np.random.shuffle(g)
        order = sorted(groups, key=lambda c: len(groups[c]))          # 小群集優先

        selected = []
        while len(selected) < k:
            for c in order:                            # 逐群輪流取樣
                if groups[c]:
                    selected.append(int(groups[c].pop()))
                    if len(selected) == k:
                        break
        return cand[selected]
```

---

## A.5　統一調用

七種策略具有一致之介面，故可於主動學習迴圈中以名稱統一調度：

```python
STRATEGIES = {
    "least_conf":     LeastConfidence,
    "entropy":        Entropy,
    "margin":         Margin,
    "coreset":        Coreset,
    "typiclust":      TypiClust,
    "badge":          BADGE,
    "cluster_margin": ClusterMargin,
}

idx = STRATEGIES[name](model, pool, labeled).query(k)
```
