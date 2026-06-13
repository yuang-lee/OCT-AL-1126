# Chapter 5 — 主動學習深入分析（規劃與追蹤）

> 本檔記錄第五章的實驗規劃、設計決策與待辦。**目前為規劃階段，尚未寫 code。**
> 與 `thesis/CLAUDE.md`（全論文追蹤）、`thesis/chapter_4/README.md`（Ch4）並列。

## 範圍與前提

- 第五章起 **聚焦三大策略各自的最佳 AL 方法**，不再跑全部七種。依 Ch4
  `al_curve_each_best.py`（最早達 88.2% 者）目前的前三名為：
  - **Uncertainty → Margin**
  - **Diversity → Coreset**
  - **Hybrid → Cluster-Margin**
  - ⚠️ 待 5-seed 資料最終定版後再鎖定（順序可能微調；以最終 al_curve 為準）。
- **既有資產：每個 AL 方法 × 每個 portion 所選的 data index 都已存檔**
  （`classification/exp_results/.../AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json`）。
  → 任何「重訓某 portion 的模型」「分析某次選樣」都可直接由 index 重建，成本低。
- 共同設定沿用 Ch4 §4.1（ResNet-18、θ² SimCLR 初始化、aug4、per-seed best-lr → mean over seeds、std ddof=1）。

---

## 5.1　主動學習超參數敏感度（b₀ 與 b）

AL 軌跡由兩個量界定：**b₀ = 初始隨機標註比例**、**b = 每輪查詢間隔**。Ch4 主結果固定
b₀ = 2.5%、b = 2.5%。本節各別變化其一（**不交叉**，以免計算量爆炸）來 justify 此選擇。

### 5.1.1　變化 b₀（固定 b = 2.5%）
- b₀ ∈ {2.5, 5, 10, 20}%。其後一律 b = 2.5% 跑同樣的三種策略到 ρ=60%。
- 目的：justify 2.5% 的初始隨機池已足夠（或找出更好的 b₀）。
- **比較須在相同總 ρ 下對齊**：b₀ 較大的軌跡「較晚才開始 AL」（第一個 AL 點落在較高 ρ）。
  以「達 target 的 ρ」「曲線下面積 / 同 ρ 之 mean acc」比較，而非看起點。

```bash
# ---- b₀ = 5% ----
B0=5  DEVICE=cuda:0 STRATEGIES="margin"  SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=5  DEVICE=cuda:1 STRATEGIES="coreset" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=5  DEVICE=cuda:2 STRATEGIES="badge"   SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh

# ---- b₀ = 10% ----
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=10 DEVICE=cuda:4 STRATEGIES="coreset" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh

# ---- b₀ = 20% ----
B0=20 DEVICE=cuda:9 STRATEGIES="margin"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:8 STRATEGIES="margin"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:8 STRATEGIES="margin"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=20 DEVICE=cuda:7 STRATEGIES="coreset" SEEDS="10" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:7 STRATEGIES="coreset" SEEDS="24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:1 STRATEGIES="coreset" SEEDS="38" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:6 STRATEGIES="coreset" SEEDS="42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:6 STRATEGIES="coreset" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下


# hybrid 改跑 cluster_margin（b₀=10、20，各 5 seeds）
B0=10 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=20 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:7 STRATEGIES="cluster_margin" SEEDS="38" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:5 STRATEGIES="cluster_margin" SEEDS="42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:5 STRATEGIES="cluster_margin" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

python3 thesis/chapter_5/plot_b0_ablation.py
```

### 5.1.2　變化 b（固定 b₀ = 2.5%）
- b ∈ {2.5, 5, 10, 20}%。
- 目的：justify 2.5% 間隔最佳。**此結論先前已大致驗證，信心較高。**
- 直覺：b 越小 → 每次用「最新、較準的模型」重新評估未標註池 → 查詢品質越高；b 越大則
  一次選太多、後段樣本是用較舊模型挑的、且批內冗餘上升 → 預期 b 越小越好（但 retrain 次數越多、計算越貴）。

### lr 注意事項
- **初始 b₀ 步 = 該 seed 的 random 選樣**（同 seed→同子集），故直接沿用 θ² cold-start 在
  該 **(b₀, seed)** 的 best lr，**免重掃**。已驗證 ρ=5/10/20 × 5 seeds 在 `cold_start_simclr` 全查得到
  （per-seed 差很多，如 ρ=5：3e-5～3e-4）。
- 機制：`run_AL.py` 新增 `--coldstart_lr_path`（指向真正的 `./classification/exp_results`），讓初始步
  的 lr 查表用真 cold-start 樹，而結果仍寫到隔離的 `ch5_b0_ablation/b0_<B0>/`。
- 後續 ρ>b₀ 步仍走 sweep + best-val（option A）。

```bash
B=10 DEVICE=cuda:0 STRATEGIES="margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5  DEVICE=cuda:0 STRATEGIES="margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:2 STRATEGIES="margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:2 STRATEGIES="margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh


B=10 DEVICE=cuda:0 STRATEGIES="coreset"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:0 STRATEGIES="coreset"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:2 STRATEGIES="coreset"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh 
B=5 DEVICE=cuda:2 STRATEGIES="coreset"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh 

B=10 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh


python3 thesis/chapter_5/plot_b_ablation.py
```

---

## 5.2　以 cold-start AL 演算法改良初始選樣（b₀ 不再純隨機）

**動機**：目前 b₀ 是「純隨機」。改用 *cold-start / low-budget AL*（不需任何標註、純用 SSL 表示空間
密度/覆蓋挑第一批）來選 b₀，期望比隨機更好的起點 → 抬升整條軌跡。注意「cold-start AL」此處指
**文獻意義的低預算主動選樣**（非 codebase 的被動 `cold_start_*` baseline）。

### 候選方法（代表性高、引用多；皆能「零標註自選第一批」）
1. **TypiClust**（Hacohen et al., *ICML 2022*）— 對 SSL 特徵分群，挑各群最典型（最稠密）點。
   **本 codebase 已實作**（`diversity_correct.py::typiclust`），可直接拿來選 b₀。低預算公認 SOTA 之一。
   → **5.2 近期就用這個當 b₀ 選樣器。**
2. **ProbCover**（Yehuda et al., *NeurIPS 2022*，"Active Learning Through a Covering Lens"）—
   把選樣視為 *Max Probability Cover*：在 SSL 嵌入空間以半徑 δ 的球覆蓋資料，貪婪挑「能覆蓋最多
   未覆蓋高密度點」者。實作輕（~60 行、吃 frozen 特徵）。**若要再加一個新 cold-start，這是首選；可晚點做。**

### ⏸ 暫緩實作（晚點再說）
- **"Making Your First Choice" / CSVAL**（Chen et al., MIDL 2023）— 忠實版需要在 **SSL 預訓練時逐 epoch
  記錄每張的對比信心 μ̂**（要動 SimCLR 訓練、重跑一次），**太麻煩 → 暫緩**。
  （若之後仍想做，可用「k-NN 密度 proxy」版規避 μ̂，但目前先不做。）
- **USL / USL-T**（Wang et al., ECCV 2022）— training-free 版雖不難，USL-T 需端到端學分群＋防 collapse 較重；
  與上者一併**暫緩**，related work 提及即可。

> 共通：都吃 frozen SSL 特徵、無需標註即可選 b₀；做成「b₀ 選樣器」即可（之後 b 步仍用 5.1 的策略）。
> **近期 5.2 範圍 = TypiClust（已有）為主，必要時加 ProbCover；其餘兩法暫緩。**

### 實驗設計
- baseline：隨機 b₀（Ch4 主結果）。對照：TypiClust / ProbCover / (Making-First-Choice 或 USL) 選 b₀。
- b₀ 大小取 5.1 結論的最佳值；後續 b 步固定用三大策略各自最佳。
- 看「換更聰明的 b₀」是否讓整條軌跡（尤其低 ρ 段）顯著上移。

---

## 5.3　所選影像之分析（量化 + 質性）

**全部可由已存的 labeled_ids index 重建，不需重跑 AL。** 預設在 **ρ=30%** 比較。

### 【優先做】混合矩陣（confusion matrix）對比
- 是的，7×7 的就叫 **confusion matrix**（列=真實類別、欄=預測類別）。
- 在 ρ=30% 下，用 **random vs 各 AL 方法** 所選 index 重訓模型，畫各自的 7×7 confusion matrix
  （在固定的 test set 上）。
- 目的：看 AL 相對 random **在哪些類別**把對角線（正確率）拉高、把哪些易混淆對的off-diagonal壓低
  → 解釋「AL 靠改善哪幾類來提升整體 acc」。
- 多 seed 取平均的 confusion matrix（或差值矩陣 AL − random）會更有說服力。

### 量化（一）：所選影像的 GT 類別分布
- 各方法在 ρ=30% 時，**已選集合 over 7 個 ground-truth label 的分布**。
- 假設：**diversity（Coreset/TypiClust）在早期應更均勻地覆蓋各類**，uncertainty（Margin）可能偏向
  幾個易混淆類別（呼應 codebase 既有的 `margin_w_statistics` top-2 類別對統計）。

### 量化（二）：剩餘未標註集的 uncertainty 分布
- uncertainty 方法（Margin）在 ρ=30% 時，對**剩餘未標註池**的 uncertainty score 分布長相
  （直方圖）→ 觀察「還剩多少高不確定樣本」、分布是否隨 ρ 變平。

### 質性：UMAP 視覺化
- 對全 train set 的 SSL 特徵做 UMAP；**不同 GT 類別 = 不同顏色的點**，
  **AL 在 ρ=30% 所選的點用紅色圈標出**。
- 目的：直觀看各策略選樣落在表示空間何處 —— diversity 應散佈、覆蓋邊角；uncertainty 應集中在類別交界。

---

## 5.4　延伸至影像分割任務（之後做）

- 嘗試把分類的 AL 觀察遷移到 2D 影像分割（`segmentation/` 既有 U-Net pipeline）。
- 細節待 5.1–5.3 收斂後再規劃。

---

## 待確認 / 開放問題
1. **b₀ 大小是否顯著影響？**（user 對此較不確定）— 見下方「討論」。先做 5.1.1 的 ablation 來定論。
2. ~~5.2 第三法選 "Making Your First Choice" 還是 USL？~~ → 兩者**暫緩**（MYFC 需動 SSL 預訓練、太麻煩）。
   5.2 近期用 TypiClust（已實作），必要時加 ProbCover。
3. Ch5 鎖定的三方法，待 Ch4 5-seed 最終 al_curve 定版後確認（目前 Margin / Coreset / Cluster-Margin）。
4. confusion matrix 與其餘 5.3 分析，是否都統一在 ρ=30%、5 seeds 平均。
