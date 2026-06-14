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

# ---- b₀ = 10% ----
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=10 DEVICE=cuda:0 STRATEGIES="cluster_margin" SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:0 STRATEGIES="cluster_margin" SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下


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

# 6/14: 加入b=20%的，這樣才更有望看到performance collapse!
B=20 DEVICE=cuda:7 STRATEGIES="margin"         SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=20 DEVICE=cuda:6 STRATEGIES="coreset"        SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=20 DEVICE=cuda:1 STRATEGIES="cluster_margin" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh


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

### 量化（一）：所選影像的 GT 類別分布 — `plot_5_3_selection_dist.py`（✅ 已實作）
- 各方法所選集合 over 7 個 ground-truth label 的分布。資料讀 `AL_simclr/labeled_ids/`
  （即主線 **b₀=2.5%、b=2.5%、5 seeds**；cumulative=該 portion 累積標記集）。
- **baseline = 整個 train set 的類別比例**（= random 的期望，analytic 無模擬雜訊），畫成灰虛線/灰 bar。
- 樣式全對齊 Ch4 AL 折線圖：色/marker（`plot_al_curve.py` GROUPS）、`figsize=(12,8)`、
  FONT 26/20/15、分組 legend（Uncertainty/Diversity/Hybrid 三欄 + Random 獨立底列）。圖無 title（caption 自寫；trend 例外，title=類別名）。輸出到 `figs/5_3_*`。
- 三種圖（`--plot`）：
  - `dist` — 各方法每類 share(%) + baseline。
  - `diff` — 相對 baseline 的偏差；`--diff pp`（百分點，預設）或 `relative`（相對%）。y 軸 `Over / Under Sampling vs. Random (%)`。
  - `trend` — 橫軸 ρ、縱軸某類 share(%) 隨 portion 變化（`--class`，預設 Normal）+ baseline 虛線；看「該類比重從哪個 ρ 開始偏離」。
- 預設策略：dist/diff = 全七種（檔名 `all`）；trend = margin/coreset/cluster_margin。明確 `--strategy` 則照給的。

```bash
# 分布圖 + 差異圖（全七種，預設 ρ=22.5%）；--portion 換比例
python3 thesis/chapter_5/plot_5_3_selection_dist.py --portion 30 --plot both

# Normal 比重 vs portion（全七種）
python3 thesis/chapter_5/plot_5_3_selection_dist.py --plot trend \
    --strategy conf margin entropy coreset typiclust badge cluster_margin
# 其餘六類同上，逐一換 --class "Eczema" / "Nevus" / "Psoriasis" / "Seborrhoeic keratosis" / "Solar lentigo" / "Vitiligo"
```

- **觀察**：除 **TypiClust** 外（density-based，貼著原分佈、Normal 維持 ~40%），其餘六法都從 ρ=5% 起把多數類
  **Normal 壓到 ~25–30%**（ρ≈12.5–17.5% 觸底）、把少數類拉高；uncertainty 因避高信心、coverage 型 diversity（k-center）因密集區少數點即覆蓋、BADGE/Cluster-Margin 兼具 → 同向。
- **與 AL 表現對照（重要）**：TypiClust 是唯一不壓 Normal 的，**也是 AL 最差**（達 88.2% 需 ρ≈42.5% vs 其餘 25–30%；ρ=60% 僅 90.5% vs ~94–95%）。但它是低預算法、用在中高預算本就吃虧（ρ=10% 時其實與他法持平），故「壓低 Normal=對」是此預算區間的相關現象，深層因是「選 informative 樣本」，類別偏移只是 symptom。
- **TypiClust 原 paper（Hacohen et al. ICML 2022, arXiv:2202.02794）對 class imbalance 的處理**：
  - §4.3.2 用 **TV distance(labeled set 類別分布, ground-truth 類別分布)** 當指標，主張 TypiClust 此距離最低
    →「queries with better class balance」；並稱「labeled set approximately class-balanced，即使選樣不看 label」。
  - §4.3.5 / App G.1.2 有在 **imbalanced CIFAR-10**（Munjal et al. 2020）測試：**低預算贏、高預算輸**。
  - **關鍵解讀（可寫進論文 discussion）**：該指標是「貼近資料真實分布」。CIFAR-10 本身平衡 → 貼近=均衡，看起來是優點；
    但在我們 **imbalanced 的皮膚資料**，同一機制 = **忠實複製多數類佔比（Normal~40%）= 不修正不平衡**，反成劣勢。
    其 paper **未**做「多數 vs 少數類 over/under-sampling」的逐類分析 → 我們這組 per-class share / trend 圖正補上此視角。

### 量化（二）：剩餘未標註集的 uncertainty 分布
- uncertainty 方法（Margin）在 ρ=30% 時，對**剩餘未標註池**的 uncertainty score 分布長相
  （直方圖）→ 觀察「還剩多少高不確定樣本」、分布是否隨 ρ 變平。

### 質性：UMAP 視覺化
- 對全 train set 的 SSL 特徵做 UMAP；**不同 GT 類別 = 不同顏色的點**，
  **AL 在 ρ=30% 所選的點用紅色圈標出**。
- 目的：直觀看各策略選樣落在表示空間何處 —— diversity 應散佈、覆蓋邊角；uncertainty 應集中在類別交界。

### 【之後做】combine TypiClust + 其他 AL（hybrid scheduling）
- 想法：**低 ρ 用 TypiClust（typical/density）選樣，高 ρ 切換到 uncertainty / BADGE 等**，
  取兩者在各自預算區間的優勢。
- **這是 TypiClust 原 paper 留下的洞**：它只觀察到「低預算用 typical、高預算用 uncertain」的相變，
  並把「低預算範圍到哪、何時切換」明白列為 future work（"...we leave for future work."），
  **沒有實作也沒提出 switching 機制**。→ 我們實作並定出切換點 = 直接回應其 future work（novelty 之一）。
- 與 §5.2（用 TypiClust 改良 b₀）相關但不同：§5.2 只換「初始批 b₀」；此處是**整條軌跡的策略排程**。
- ⚠️ caveat：本 codebase 的 TypiClust 是 **warm-start、用當前 finetuned 模型的 backbone 特徵**，
  已偏離原版「frozen SSL 特徵 + 零標註自選首批」；做 hybrid 前需決定是否補一版忠實的 frozen-SSL TypiClust。
- **狀態：留到之後做（先完成上面的 confusion matrix / 類別分布 / UMAP 量化分析）。**

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
