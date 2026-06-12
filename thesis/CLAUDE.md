# CLAUDE.md — 論文實驗追蹤 (Thesis experiment tracker)

本檔追蹤碩士論文（光電所）各章節**需要的實驗、目前資料狀態、與待跑清單**。
論文 PDF：`thesis/draft/Yu-AngLee-Thesis-0427.pdf`（docx 本體請存一份到 `thesis/draft/`，目前只有 lock 暫存檔）。
彙整/對比工具：`thesis/chapter_4/aggregate_results.py`（從 codebase 撈真實數字、與論文數字對比、標出覆蓋缺口）。

> 重要：論文中的數字可能落後於 codebase。**永遠以 `aggregate_results.py` 撈出的 codebase 數字為準**，不要相信 PDF 裡的數字直到對比過。

---

## 論文骨架
六章。Ch1 緒論、Ch2 OCT 與皮膚、Ch3 方法與資料集、**Ch4 提升分類標註效率（本檔重點）**、Ch5 主動學習深入分析、Ch6 結論。
Ch2、Ch3 已完整；Ch4 進行中；Ch5、Ch6 大綱階段。

---

## Chapter 4 — 實驗設置（canonical，取自 §4.1，作為所有 Ch4 實驗的共同設定）

- **模型**：ResNet-18。資料 2541 張 → train 2032 (75%) / val 254 / test 255。不論 ρ 為何，val/test 一律用完整集。
- **下游分類 head**：原 fc 換成 `W_FC ∈ ℝ^{7×512}`（七類）。損失 = cross-entropy，optimizer = Adam。
- **下游訓練超參**：batch size 16、epoch 20、初始 lr 線性衰減至 0。
  - ⚠️ **但每個 ρ 都會掃多個 lr、取最佳**（這就是「有 tune learning rate」的證據；資料結構 `{aug:{ρ:{lr:[runs]}}}`）。論文沒寫 tuning（老師不信其重要性），但 codebase 確實有做。報告時 official 慣例：**best-lr per portion**。
  - **θ¹/θ² finetune 精簡 lr 網格 + seed（2026-06-08，user 定）**：見 `run_4_3_simclr_finetune.sh`。
    - lr：2.5/5/10=`3e-5 5e-5 1e-4 3e-4`；20/30/40=`5e-5 1e-4 5e-4`；50–90=`1e-4 5e-4`；100=`1e-4 5e-4 7e-4`。依「預訓練 init 最佳 lr 在 {5e-5…5e-4}、用不到 7e-6/1e-5」精簡。⚠️ **刻意不含 3e-4**（其實是 ρ=20/30/50/60/70 實測最佳）→ user 為求快接受 mid portion 可能低 ~1–2%。
    - seed：ρ<70=5 seeds；**ρ≥70（70/80/90）=3 seeds（10,24,42）**；ρ=100=1 seed。
    - 全 θ¹ 約 900→440 trainings。θ_rand/θ_ImageNet 已完成、不受影響（用舊的完整網格/5 seeds）。
- **三種初始化 θ₀**：
  - `θ_rand` = Kaiming 隨機初始化
  - `θ_ImageNet` = ImageNet 預訓練權重
  - `θ_SimCLR`，再細分兩種**「SimCLR 的起點」**（本論文 novelty，少有文獻比較）：
    - **θ¹_SimCLR** = 從 **random init** 出發做 SimCLR（`run.py -a resnet18_random`）
    - **θ²_SimCLR** = 從 **ImageNet init** 出發做 SimCLR（`run.py -a resnet18`，預設）
    - 研究假設：**θ² > θ¹**（從 ImageNet 出發再 SimCLR 更好）。
  - 三種情境下 `W_FC` 皆隨機初始化。
- **SimCLR 預訓練超參**：Adam、weight decay 1e-4、初始 lr 2e-4、cosine annealing 至 0、溫度 τ=0.07、out_dim 32、2 層 MLP projection（512→512→ReLU→32）。
  - 掃描：`BS ∈ {16,32,64,128,256}`、`E ∈ {10,20,50,100,200,500}`。
  - ⚠️ **論文 §4.1 筆誤**：寫成 `E ∈ {10,25,50,...}`，實際 codebase 與熱力圖是 **20** 不是 25。**待改 PDF**。
- **資料增強**：Ch4 所有 4.3 / 4.4 的 finetune **一律用 aug4（4x = HF+VF+HFV）**，依 4.2 結論為最佳。程式預設 `--aug_factor 4`（run_first_iter_simclr.py / run_AL.py），結果存在 JSON 的 `"aug4"` key 下。早期 `aug2_*/aug3/no_aug/*jitter*` 只是 4.2 比較用的零星資料，非主線。
- **標註比例 ρ** ∈ {2.5, 5, 10, 20, 30, ..., 100}%。`|D_train^L| = ρ·|D_train|`。
- **重複次數慣例**：
  - 若 D_train^L 為隨機挑選 → **隨機挑 5 次，每次訓練 3 回**，報 5 次的平均±標準差（反映子集挑選變異）。
  - 否則 → 重複訓練 3 次。
  - ⚠️ **ρ=100 是「否則」情形**：全集被選入、與 seed 無關（`random.sample(全集,全集)` 不論 seed 都相同）→ **只用單一 seed（42）× 3 runs**，不跑 5 seeds。finetune 腳本已對 ρ=100 自動單 seed；θ_rand/θ_ImageNet/θ² 在 ρ=100 也都是 n=3。
- **分卡負載平衡慣例**：finetune 腳本切 portion band 時，train 時間 ∝ portion 大小，**越大（越後面）的 portion 一張卡放越少個**（如 θ¹ finetune = low{2.5..50} 7 個 / mid{60,70,80} 3 個 / high{90,100} 2 個）。
- **重跑安全**：`run_first_iter_simclr.py` 的 `check_existing_results`（max_runs=3，per aug×portion×lr）已完成會 raise，被腳本 `|| true` 跳過、未滿會補到 3 → 腳本可安全重跑（會印 `Experiment already completed!` 屬正常）。
- **硬體**：單機單卡，NVIDIA RTX 3090 或 A6000。PyTorch 2.5.1。
- **GPU 對應**：`cuda:N` 為邏輯編號，實體對應見 repo 根的 `gpu_map.md`。

---

## Chapter 4 — 逐節狀態（已用 aggregate_results.py 驗證 2026-06-08）

### 4.2 資料增強 — ✅ 完成
Table 4-1 + 圖齊全（ρ 2.5→100，flip 策略 w/o, HF, VF, HF+VF, HF+VF+HFV，θ₀=ImageNet）。
資料：`classification/exp_results/classification_hard/cold_start_imagenet/`。

### 4.3 自監督學習 — ⚠️ 未完成（最大塊）
Table 4-2 四欄狀態：
- `θ_rand`：✅ 全 ρ（codebase 與論文差 <1.5%，差異來自 lr 挑選慣例，需統一）。資料 `cold_start_random/`。
- `θ_ImageNet`：✅ 全 ρ（同上）。資料 `cold_start_imagenet/`。
- `θ²_SimCLR`（ImageNet→SimCLR，best cfg = lr0.0002/bs256/ep500）：
  - **多 seed 僅 ρ=2.5/10/100**（= 論文已填的 55.90/71.03/96.17）。
  - 中間 ρ=5,20,…,90 **只有 seed42**（藏在彙整檔 `cold_start_simclr/random42_bs16.json`，是 4.3 曲線圖的來源），**缺多 seed 標準差**。
  - **待跑**：θ² best-cfg 在 ρ∈{5,20,30,40,50,60,70,80,90} × 5 seeds finetune。
- `θ¹_SimCLR`（random→SimCLR）：❌ **完全沒有資料，連 checkpoint 都沒 pretrain**。
  - **決策（2026-06-08）**：θ¹ **不做 bs×ep ablation**，固定 **bs256/ep500**（直接用最好的）。
  - ⚠️ **但 lr 不照搬 θ² 的 2e-4**：random init 的最佳 SimCLR lr 與 ImageNet init 不同（from-scratch 通常需較大 lr），直接用 2e-4 會讓 θ¹ 訓練不足、使「θ²>θ¹」贏得不乾淨。
  - **待做**：① `run.py -a resnet18_random` 在 bs256/ep500 下掃 `simclr_lr ∈ {1e-4, 2e-4, 4e-4}`（3 個 pretrain）→ 用 linear-eval 或下游@100% 挑最佳 → 定為 θ¹ 正式 checkpoint；② 拿該 checkpoint 下游 finetune 全 ρ × 5 seeds。
  - 若 θ¹ 在 ep500 明顯未收斂，考慮加跑 ep1000。
  - ⚠️ **需小程式修改**：`run_first_iter_simclr.py:57` 的 `build_simclr_path` 寫死 `resnet18_simclr_` 前綴，且結果都寫 `cold_start_simclr/`。要加 `--simclr_init {random,imagenet}` 維度：載入對應 `resnet18[_random]_simclr_*.pkl`，並把結果寫到分開資料夾（如 `cold_start_simclr_randinit/`）以免覆蓋 θ²。
- **bs×ep 下游熱力圖**（θ²，固定 simclr_lr=2e-4）：⚠️ **要補完整**（目標：證明 bs↑/epoch↑ → 下游↑ 的單調性；目前 ρ=100 只有反對角帶，看不出趨勢）。兩張都要：
  - **ρ=100%**：缺 12/30 格 = bs{16,32}×ep{100,200,500} + bs{128,256}×ep{10,20,50}。seed42 單 seed（與現有論文圖一致）。
  - **ρ=10%**：缺 5/30 格 = bs128×ep{20,50,100,200,500}（此 portion 用 5 seeds）。ρ=10 現有格已清楚呈現單調趨勢。
  - ✅ **30 個 SimCLR checkpoint 全已存在**（`SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs*_ep*.pkl`）→ **不需 pretrain，純下游 finetune**。
  - **待跑**：`./thesis/chapter_4/run_4_3_heatmap_fill.sh`（薄包裝呼叫既有 `simclr_meta.sh`）。
  - 驗證覆蓋：`python3 thesis/chapter_4/aggregate_results.py --heatmap`。
  - 註：θ¹（random→SimCLR）理想上也想要同樣兩張熱力圖做對照，但需先 pretrain θ¹ checkpoints。
  - ⚠️ **方法學 caveat（lr–bs scaling）**：現有熱力圖對所有 batch size **固定 SimCLR lr=2e-4**，並未隨 bs 做 linear scaling。
    - 實證：bs256 時 2e-4 (96.17) > 4e-4 (94.90)，與線性 scaling 預測「大 bs 要大 lr」**相反**。
    - 原因：本研究 SimCLR 用 **Adam**（非原論文 LARS/SGD）+ 小資料集 → 最佳 lr 對 bs 曲線平坦，linear scaling 不適用。固定 2e-4 接近各 bs 最佳。
    - **決定**：採 (B) 固定 2e-4 + 論文加說明（2026-06-08）。
  - ⚠️ **seed 覆蓋不均（重要）**：ρ=100 熱力圖**整張僅 seed42 × 3 runs**（無跨 seed std）；ρ=10 為混合（bs16 列、bs256 高 epoch = 5 seeds×3 runs；bs64 列、bs128、bs32 高 epoch 僅 1 seed）。解讀單 seed 格要小心（如 ρ=10/bs64/ep500=73.6 是單 seed artifact）。
  - **interaction 分析計畫（待跑）**：要證「大 bs pretraining 的好處隨 downstream portion 增加而變大」需 portion×bs 交互。最小乾淨設計：固定 ep=500、固定 5 seeds，把 bs∈{16,32,64,128,256} 在 ρ∈{10,30,100} 全填同格同 seed，再比各 portion 的 bs→acc 斜率。
  - **30% 熱力圖（user 要求，之後做）**：ρ=30 是上述 interaction 的中間點。先記著，補完 10/100 缺口後再做。
  - 目前資料**方向上**支持「大 bs 對大 portion 較有幫助、對小 portion 幾乎無感」，但覆蓋不均 + 單 seed 使其尚不能成定論。

### 4.4 主動學習 — 協定（option A，已實作）
**run_AL.py 正式協定（2026-06-08，user 定）：**
- **Random seed**：每個 `--seed` = 一條獨立 AL 軌跡，**初始 labeled pool 由該 seed 隨機選取**（[run_AL.py](../classification/run_AL.py) `random.seed(args.seed)`）。跑 **5 個 seed（10,24,38,42,57）→ 5 條軌跡 → mean±std 取自 5 seeds**。
- **起始/間隔**：portion_start=**5%**、interval=**2.5%**、end=62.5（跑到 60）。即 ρ=5,7.5,…,60（23 點）。
- **Learning rate = option A（sweep + best-val 選取）**：`--lr_schedule sweep`（預設）。每個 portion 對候選 lr（`lr_grid_for(portion)`：ρ<20→`5e-5 1e-4 3e-4`，否則 `1e-4 3e-4 5e-4`；可用 `--lr_grid` 覆寫）**各自 fresh-init 訓練**，**用 validation loss 最低的 model 當「選取器」去選下一批**（挑 lr 用 val 而非 test，避免 test leakage）。每個 lr 的 test_acc 都存進 JSON（aggregate 取 best-lr 一致）。
  - 為此 `train_eval.py` 的 `train_model` 多回傳第 3 值 `best_val_loss`（caller run_first_iter*.py 已改成忽略）。
  - 其他模式：`coldstart`（單一查表 lr，option B）、`fixed`（單一 --lr）。
- **初始步 ρ=2.5% 的 lr**：與 Random baseline 一致——用該 **seed 在 cold-start(θ²) 的 best-lr**（同 seed→同 2.5% 子集，故 AL 起點 = Random 起點）。各 seed 差很多（10→5e-5, 24→1e-4, 38→5e-5, 42→5e-4），所以必須 per-seed。後續 ρ>2.5% 才 sweep（AL 選樣改變 → optimal lr 不同）。seed57 cold-start 無 2.5% → 退回 sweep。`run_AL.py: coldstart_best_lr()`。
  - **`--coldstart_lr_path`（2026-06-12 加，backward-compat，預設沿用 `--exp_path`）**：把「初始步查 per-seed cold-start best-lr」的根目錄與「寫結果」的 `--exp_path` 解耦。用途＝結果寫到隔離樹（如 Ch5 b₀ ablation 的 `ch5_b0_ablation/b0_<B0>/`）時，仍能去**真** cold-start 樹（`./classification/exp_results`）查初始 lr，免重掃。主 4.4 不傳此參數、行為不變。Ch5 `run_5_1_b0_ablation.sh` 會自動帶（初始 b₀ 池＝該 seed random 子集 → 用 cold-start(b₀,seed) best-lr）。
- **每 portion 不重複 3 次**：sweep 內每個 lr 訓練一次；變異來自 5 個 seed。
- **Random/passive baseline**：`--AL_strategy random`（已加為合法策略，每步隨機選），跟其他策略同協定一起跑。
- **全論文彙整慣例（2026-06-10，user 定）= per-seed best-lr → mean±std over seeds**：每個 seed 先以「**該 seed 自己 runs 的平均**」挑出自己的 best lr，得每 seed 一個 representative acc，再對 seeds 取 mean/std。4.2/4.3/4.4 **是同一個方法**，只差每 seed 的 run 數（4.2/4.3 約 3 runs→用 3-run 平均挑 lr；4.4 現在 1 run→就是該單跑）。**不是 pooled**（把所有 seed 的 runs 合併挑單一 lr）。per-seed 比 pooled 樂觀略高（例：coreset ρ=12.5 per-seed=77.1 vs pooled=74.9）。
  - 程式狀態（皆 ✅ per-seed-best，2026-06-10 統一）：4.2 `data_aug/plot_all.py`（Cross-JSON；terminal = `print_summary_table`）；4.3 `aggregate_results.py::pool_seed_files()`（已改寫成 per-seed-best）→ `plot_portion_curve.py` / Table 4-2 同步；4.4 `plot_al_curve.py::_per_seed_best_curve()`。
  - 改動後 4.3 數字略升（per-seed > pooled），例 θ²：ρ=5 63.4→64.5、ρ=30 83.8→84.5（seed 一致或單 seed 的點不變）。heatmap(`plot_simclr_heatmap.py`)本來就是 per-seed（per seed selected-lr → over seeds），不受影響。
  - **std 一律 ddof=1（樣本標準差，2026-06-10 統一）**：4.2/4.3/4.4 的表格與圖 std 帶都用 ddof=1（5 個 seed 是樣本、估計母體變異）。4.2 原本是 ddof=0 已改（`plot_all.py::_sstd`）。**ρ=100（單 seed，seed-independent）**：std 改用「該 seed best-lr 的 runs」的 ddof=1（`pool_seed_files` 單 seed 分支 + 4.2 SPECIAL_RHO_100），故 4.2 4x 與 4.3 θ_ImageNet 在 ρ=100 也一致（≈±1.0，非 0）。
- **labeled id 匯出**：每 portion 的 selected/cumulative + lrs_swept 存到 `AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json`（reproduce + Ch5 視覺化）。
- **執行**：`thesis/chapter_4/run_4_4_active_learning.sh`（6 策略 random+conf+entropy+margin+coreset+badge × 5 seeds，θ² best ckpt 初始化）。畫圖：`thesis/chapter_4/plot_al_curve.py`（Random 灰虛線 + 5 策略，2.5 interval）。
- ⚠️ 現有 `AL_simclr/*.json` 是**舊資料**（seed42、固定 lr、舊 ckpt）→ 依新協定**重跑**。
- **AL 演算法 bug 全紀錄（2026-06-08，已查證 paper；正確版在 `AL_strategy/{diversity,hybrid}_correct.py`，舊版保留為 `*_wrong.py`）**：
  conf / entropy / margin 三個 uncertainty 方法**本來就正確**。Coreset 與 BADGE 各踩了多個雷：
  1. **特徵抽取抽錯維度（最嚴重，coreset+badge 都中）**：`Sequential(*list(model.children())[:-1])` 對 `ResNetSimCLR` 失效——它的 children **只有一個 `.backbone`**，所以 `[:-1]` 變空的 = identity → 直接吐**原始影像(3×H×W≈1.8M 維)**當「特徵」。後果：coreset 在原始像素空間算距離、badge 用原始影像當梯度嵌入的 feature（**完全錯的特徵空間**），且 coreset 算 `[Nu,Nl,1.8M]` 距離 → **OOM(677 GiB)**。修正：`net = model.backbone if hasattr(model,'backbone') else model` 再 `Sequential(net.children()[:-1])` → 正確 512 維。coreset 距離也改 matmul 免 3D broadcast。
  2. **Coreset 沒 condition 在已標註集**：舊版只收未標註、從隨機未標註點開始 furthest-first，**沒看已標註資料**（偏離 Sener & Savarese 2018：應以「所有已標註樣本」為初始中心）。修正：`coreset()` 加收 `labeled_idx`，min_distances 由「到最近已標註點」初始化；`run_AL.py` 呼叫處傳 `label_idx`。
  3. **BADGE 距離公式錯**：舊版 `distance_vectorized` 範數項把 i 與 center 的範數交叉相乘（自身距離≠0，已數值證實），k-means++ 用錯距離。正確：`‖g_i‖²=‖mp_i‖²·‖emb_i‖²`（同點兩範數相乘）。
  - ⚠️ **所有舊 coreset/badge 結果都是 buggy 版（錯特徵 + 上述）產生的 → 一律刪除、用正確版重跑。** 已查 [BADGE arXiv:1906.03671] / [Coreset ozansener ICLR2018]。
- **新增兩個 AL 策略（2026-06-10，為了找「能否再提升」）**：依官方碼/論文實作，都重用「進 `model.backbone` 抽 512 維」特徵。
  - **TypiClust**（Diversity，Hacohen et al. ICML 2022，官方 repo `avihu111/TypiClust`）：對 (已標註+未標註) 特徵 KMeans（`n_clusters=min(|L|+budget,500)`），群依 (已標註少↑, 大小↓) 排序、排除 size<5 小群，每群挑 **typicality=1/(K_NN=20 平均距離+1e-5)** 最高的未標註點。`diversity_correct.py::typiclust`。**定位：低預算(小 ρ)可能真的贏過 coreset/BADGE（公認低預算 SOTA）；高預算反而不一定。**
  - **Cluster-Margin**（Hybrid，Citovsky et al. NeurIPS 2021）：取 margin 最小的 `k_m=10·k_t` 候選 → 候選特徵 **L2-normalize** → HAC(average linkage, `distance_threshold=ε`) → 群按大小升冪 round-robin 每群隨機取一，湊 k_t。**ε 論文未明定** → 本實作取「候選兩兩距離中位數 × `EPS_FRAC=0.5`」（尺度自適應、可調）。`hybrid_correct.py::cluster_margin`。**定位：比 BADGE 輕量、效果相近（非更強），當對照組；論文勿寫「比 BADGE 強」。**
  - 接線：`run_AL.py` choices/import/dispatch 已加（**未動既有 6 策略**）；`plot_al_curve.py` GROUPS：Diversity 加 TypiClust(淺綠)、Hybrid 加 Cluster-Margin(淺紅)，legend 拆成 Uncertainty|Diversity|Hybrid 三欄。（修了個解析 bug：`f.split("_")[0]` 對 `cluster_margin` 會切錯 → 改 `split("_seed")[0]`。）
  - **TypiClust 是「cold-start / low-budget AL」家族，與本論文 Ch3 的 uncertainty AL 是不同範式**（user 2026-06-10 指出）。注意「cold start」一詞有兩義：① codebase 的 `cold_start_*` = 被動 baseline（隨機 ρ%、無 AL，4.3 portion 曲線）；② 文獻的 cold-start AL = 處理「極少/零標註下如何選樣」的一類主動方法（TypiClust/ProbCover）。TypiClust 屬 ②，純用表示空間密度選樣、不看模型 uncertainty；原版用 **frozen 自監督特徵**且能**無標註自選第一批**。
  - **目前決定（option A，維持現狀）**：TypiClust/Cluster-Margin 與所有策略一致——**初始 2.5% 一律 random**（同 seed→同初始池，公平比較 + 沿用 cold-start best-lr），且 TypiClust **用當前 finetuned 模型的 backbone 特徵**（非 frozen SimCLR）。Cluster-Margin 本來就需要 warm-start（margin 要有 model）。
  - **【Ch5 待探討】cold-start AL 的選擇能否「搭配其他策略」進一步提升 performance**（user 想在第五章寫）：可做的 bonus 實驗——(i) TypiClust 改用 **frozen SimCLR 特徵 + 自選初始批**（忠於原版，發揮低預算優勢）；(ii) **混合排程**：低 ρ 用 TypiClust（density）、高 ρ 切換到 uncertainty/BADGE（呼應原 paper「opposite strategies suit high/low budgets」）；(iii) 比較「random 初始 vs TypiClust 自選初始」對後續軌跡的影響。主結果仍走 option A，這些當 Ch5 深入分析。

### 4.5 綜合比較各項策略 — factor ablation（設計定案 2026-06-12，待跑 4 arm）
**目的**：4.2–4.4 是逐一疊加（AL 只做在 aug4+θ² 上）；4.5 拆解 Aug(4x)／Init(θ²)／AL 三策略**各自**貢獻。
- **兩個 Table，rows ρ=10/30/50%**：Table 1 一次只開一個（Data Aug | Weight Init | AL | All Three）；Table 2 一次只關一個（w/o 各項 | All Three）。「關」= no_aug / ImageNet / passive random。
- **AL 代表策略 = `margin`（user 定案 2026-06-12：4.4 所有方法中最好）**。要換策略時三個 AL arm 加 `STRATEGY=xxx` 重跑（檔名含策略名可並存；init_only 與策略無關）。
- **7 cell 對照（Aug, Init, AL）**：3 個用主實驗既有資料——aug_only=(aug4,ImageNet,✗)=4.2、wo_al=(aug4,θ²,✗)=4.3 θ² best cfg、all_three=(aug4,θ²,margin)=4.4；另外 (no_aug,ImageNet,✗) 全關 baseline 也已在 4.2 的 `no_aug` key（aggregate 會印參考列）。**待跑 4 arm**：
  - `init_only`=(no_aug,θ²,✗)：cold-start **只跑 ρ=10/30/50** 三點，5 seeds×3 runs，lr 網格沿用 4.3 精簡版。
  - `al_only`=(no_aug,ImageNet,margin)、`wo_aug`=(no_aug,θ²,margin)、`wo_init`=(aug4,ImageNet,margin)：AL 軌跡 **ρ=2.5→50（interval 2.5，20 點）**，同 4.4 option A（sweep `3e-5 5e-5 1e-4 3e-4`、每 lr 1 run、best-val 選取器、5 seeds）。
- **結果隔離（重要，user 要求勿與主實驗混）**：全部存 `classification/exp_results/chapter4_5_ablation/{arm}/classification_hard/...`（靠 `--exp_path` 分樹，程式零修改）。
- **執行**：`ARM=init_only|al_only|wo_aug|wo_init DEVICE=cuda:N [SEEDS=...] ./thesis/chapter_4/run_4_5_ablation.sh`。重跑安全：AL arm 對已存在 (strategy,seed) JSON 跳過（`FORCE=1` 接續）；cold-start arm 由 check_existing_results（3 runs 滿）擋。
- **檢視**：`python3 thesis/chapter_4/aggregate_4_5.py`（直接印兩個 Table + 全關 baseline，per-seed best-lr → mean±std ddof=1，缺 cell 標 NA）。
- ⚠️ **協定微差（已知、可接受）**：新 AL arm 的初始 2.5% 在獨立 exp_path 查不到 cold-start 參照 → 一律退回 sweep；all_three（4.4 主跑）2.5% 用 per-seed cold-start best-lr。三個新 arm 內部一致，論文不需提。
- ⚠️ lr 網格各 cell 沿用其「家族」慣例（4.2 寬網格 / 4.3 精簡網格 / 4.4 sweep 網格），跨 cell 不完全相同——與 4.2–4.4 主結果同一狀況，比較時心裡有數即可。
- 現況（2026-06-12 aggregate 實測）：3 個既有 cell 有數字（all_three@50 暫 n=4，等 4.4 margin 跑完）；4 個新 arm 全 NA 待跑。

---

## 待決定 / 待確認清單
1. θ¹_SimCLR 用哪個（些）SimCLR cfg？建議至少鏡像 θ² 的 best（lr2e-4/bs256/ep500）做 apples-to-apples；是否也要 θ¹ 的完整 bs×ep 熱力圖？
2. 是否同意做 `--simclr_init` 的程式修改（θ¹ 必需）。
3. 4.4 是否加 random baseline。
4. θ_rand/θ_ImageNet 的 lr 挑選慣例要統一（per-seed best 再平均 vs pooled best）——影響論文數字小數點。

## 已知 PDF 待修
- §4.1 SimCLR epoch 集合 `{10,25,...}` → 應為 `{10,20,...}`。
- Table 4-2 θ²_SimCLR 中間 ρ、θ¹_SimCLR 整欄待填（待實驗）。
