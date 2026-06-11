# 新增兩個 AL 演算法整理（給 Ch3 介紹用）

本文件整理我們新加入的兩個主動學習（Active Learning, AL）策略：**TypiClust**（Diversity 類）與
**Cluster-Margin**（Hybrid 類）。包含論文出處、核心想法、**我們實際 implement 的公式與步驟**、
以及我們採用的超參數/工程選擇。給 coworker 依此擴充碩論 **Chapter 3** 的方法介紹。

> 共同前提（兩者皆同）：backbone = **ResNet-18**；特徵 = **penultimate layer 的 512 維向量**
> （SimCLR 包裝模型須進 `model.backbone` 抽，否則會誤抽到原始影像；plain resnet 則 `model`）。
> 每一步要選的標註數記為 **k_t**（= 該 portion 需新增的 labeled 數）。

---

## 1. TypiClust — Diversity（low-budget / cold-start AL）

- **論文**：Guy Hacohen, Avihu Dekel, Daphna Weinshall, *"Active Learning on a Budget:
  Opposite Strategies Suit High and Low Budgets"*, **ICML 2022**. arXiv:2202.02794.
- **官方程式碼**：<https://github.com/avihu111/TypiClust>
- **類別/定位**：diversity / representation-based；專為**低標註預算**設計（low budget），
  在標註極少時常勝過 uncertainty 與 coreset。屬「cold-start AL」家族（不依賴模型不確定性，
  靠表示空間的密度選樣）。

### 核心想法
低預算下模型很弱、其 uncertainty 不可信；因此改成在**特徵空間**挑「**典型（typical）且具代表性**」
的樣本——即位於**高密度區域**、且能**覆蓋尚未被標註的群集**的點。

### 我們 implement 的步驟與公式
1. 對 **(已標註 ∪ 未標註)** 的 512 維特徵做 **K-means** 分群，群數
   `K = min(|L| + k_t, 500)`（|L| = 目前已標註數）。
2. 每個群統計：大小、含多少已標註點。排除大小 < `MIN_CLUSTER_SIZE = 5` 的小群；
   其餘群依 **(已標註數 ↑, 群大小 ↓)** 排序（優先處理「還沒被標註覆蓋、又大」的群）。
3. 對某點 x，定義 **typicality（典型度）**：
   $$\mathrm{Typicality}(x) = \frac{1}{\frac{1}{K}\sum_{x_i \in \mathrm{KNN}(x)} \lVert x - x_i \rVert_2 \; + \; \varepsilon}$$
   其中 K = `K_NN = 20`、ε = 1e-5。**到最近鄰平均距離越小 → 密度越高 → typicality 越大。**
4. **Round-robin** 走訪排序後的群：在每個群的**未標註成員**中，挑 typicality 最高者加入標註集，
   標記已選，直到湊滿 k_t。

### 我們的超參數選擇
- `K_NN = 20`、`MIN_CLUSTER_SIZE = 5`、`MAX_NUM_CLUSTERS = 500`（皆同官方預設）。
- 群數 `K = min(|L|+k_t, 500)`（同官方）。≤50 群用 `KMeans`，否則 `MiniBatchKMeans`。
- 程式：`classification/AL_strategy/diversity_correct.py::typiclust`。

### 一句話寫法（給 Ch3）
> TypiClust 在自/監督特徵空間以 K-means 分群，於每個尚未被覆蓋的群中挑選 typicality（KNN 密度倒數）
> 最高的代表點，藉此在低標註預算下選出兼具「代表性」與「多樣性」的樣本。

---

## 2. Cluster-Margin — Hybrid（uncertainty × diversity）

- **論文**：Gui Citovsky, Giulia DeSalvo, Claudio Gentile, Lazaros Karydas, Anand Rajagopalan,
  Afshin Rostamizadeh, Sanjiv Kumar, *"Batch Active Learning at Scale"*, **NeurIPS 2021**.
  arXiv:2107.14263.（Google Research；無官方碼，社群實作見 github.com/FNTwin/cluster_margin。）
- **類別/定位**：hybrid（uncertainty + diversity）。主打**大批量、可擴展**（百萬級）；
  以**更低成本**達到與 BADGE 相近的效果（注意：是「相近/更省」，**不是更準**）。

### 核心想法
先用 uncertainty（margin）篩出一批最不確定的候選，再用**階層式分群（HAC）**對這批候選做多樣化，
最後**跨群輪流（round-robin）**取樣，兼顧「不確定」與「不重複」。

### 我們 implement 的步驟與公式
1. 對每個未標註 x 算 **margin（信心差）**：
   $$m(x) = p_{(1)}(x) - p_{(2)}(x)$$
   即 softmax 後**最大與次大類別機率之差**（越小 = 模型越不確定）。
2. 取 margin **最小的** `k_m = 10 · k_t` 個為候選集（最不確定者；上限為未標註池大小）。
3. 對候選的 512 維特徵做 **L2 normalize**，再做 **HAC（Agglomerative Clustering，average linkage，
   歐氏距離）**，分群門檻 `distance_threshold = ε`。
   - average linkage 群間距離：$d(A,B) = \frac{1}{|A||B|}\sum_{a\in A, b\in B}\lVert a-b\rVert_2$，
     合併至 $d(A,B) > \varepsilon$ 為止。
4. 候選依其所屬群分組，**群按大小升冪排序**；**round-robin** 每群隨機取一個、循環（跳過已清空的群），
   直到湊滿 k_t。

### 我們的超參數選擇
- `k_m = 10 · k_t`（同論文 Open Images 設定）。
- **ε（HAC 門檻）**：論文只說「predefined」未給值。我們採**尺度自適應**：
  `ε = median(候選兩兩距離) × 0.5`（先 L2-normalize，故距離落在 [0,2]）。可調參數 `eps_frac`（預設 0.5）。
- 程式：`classification/AL_strategy/hybrid_correct.py::cluster_margin`。

### 一句話寫法（給 Ch3）
> Cluster-Margin 先以 margin（前兩大類別機率之差）選出最不確定的候選，對其特徵做 average-linkage 階層分群，
> 再跨群輪流取樣，以低計算成本同時達成不確定性與多樣性。

---

## 兩者在本論文中的定位（建議寫法）
| 方法 | 類別 | 對照的既有方法 | 我們的期待/定位 |
|---|---|---|---|
| **TypiClust** | Diversity（low-budget） | Coreset（geometry/furthest） | **低 ρ 可能真的贏過** coreset/BADGE（density vs geometry 對比）|
| **Cluster-Margin** | Hybrid | BADGE（gradient embedding + k-means++）| **更輕量、效果相近**的對照（勿寫「比 BADGE 強」）|

## 引用（BibTeX 關鍵字）
- TypiClust：Hacohen et al., ICML 2022, arXiv:2202.02794.
- Cluster-Margin：Citovsky et al., NeurIPS 2021, arXiv:2107.14263.
- （對照）Coreset：Sener & Savarese, ICLR 2018；BADGE：Ash et al., ICLR 2020.
