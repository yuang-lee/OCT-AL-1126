# Chapter 4 — 實驗執行手冊

本資料夾放 Chapter 4「提升標註效率」**還缺的實驗腳本**與**結果彙整工具**。
逐節資料狀態、實驗設置、待決定事項見 [`../CLAUDE.md`](../CLAUDE.md)（論文實驗追蹤主檔）。

- 所有 `.sh` 從 **repo 根目錄**執行（腳本內部會自己 `cd` 到正確位置）。
- `DEVICE=cuda:N` 一律明確指定；實體 GPU 對應見 repo 根 `gpu_map.md`。
- 這些是昂貴 GPU 工作，請自行啟動並控管 GPU 分配。

---

## 0. 彙整 / 對比工具（隨時可跑，不需 GPU）

```bash
python3 thesis/chapter_4/aggregate_results.py            # Table 4-2 各 init × ρ + AL + 熱力圖，與論文數字對比
python3 thesis/chapter_4/aggregate_results.py --heatmap  # 只看 θ² bs×ep 熱力圖覆蓋（ρ=100 與 ρ=10）
python3 thesis/chapter_4/aggregate_results.py --theta1   # 只看 θ¹ 結果 + θ¹ vs θ² 對照
```
慣例：每個 ρ 在掃過的多個下游 lr 中取 **best-lr**，報 mean±std（×100）。
**永遠以此工具的 codebase 數字為準，不要相信 PDF 裡的舊數字。**

---

## 建議執行順序

優先序：**A（θ¹，4.3 最大缺口、論文 novelty）** → **B（θ² 熱力圖補完）** → **C（θ² 中間 portion 多 seed）** → **D（4.4 AL，待 run_AL.py 改）**。

---

## A. θ¹_SimCLR（random→SimCLR）— 4.3 核心

目的：證明 **θ²（ImageNet→SimCLR）> θ¹（random→SimCLR）**。填 Table 4-2 的 θ¹ 欄 + portion 曲線。
固定 bs256/ep500（不做熱力圖）。結果寫到 `classification/exp_results/classification_hard/cold_start_simclr_randinit/`。

```bash
# A1. 預訓練 random-init SimCLR，掃 3 個 pretraining lr（可三卡並行）
DEVICE=cuda:3 SIMCLR_LRS="5e-5" ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
DEVICE=cuda:0 SIMCLR_LRS="1e-4" ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
DEVICE=cuda:1 SIMCLR_LRS="2e-4" ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
DEVICE=cuda:2 SIMCLR_LRS="4e-4" ./thesis/chapter_4/run_4_3_theta1_pretrain.sh
#   → 產出 SSL/simclr/ckpt/resnet18_random_simclr_lr{...}_bs256_ep500.pkl

# A2.（已跳過）原本要 finetune 挑 θ¹ 最佳 pretraining lr。
#    決定：θ¹ 直接用 lr=0.0002，與 θ² 完全相同的 SimCLR recipe（lr2e-4/bs256/ep500），
#    只差 init（random vs ImageNet）→ 受控比較，θ²>θ¹ 的差異乾淨歸因於起點。
#    （注意：論文勿寫「因 2e-4 的 SimCLR 準確率最好」——θ¹ 的 contrastive acc 其實 4e-4 最高；
#      理由要寫「相同設定、只差初始化」。）

# A3. 用 meta script 跑 θ¹ 全 portion（INIT=theta1；一卡並行 MAX_PAR=3 吃滿 GPU）
#     越大 portion 越花時間 → 一卡放越少個；cuda:N≠實體GPU，見 gpu_map.md
INIT=theta1 PORTIONS="2.5 5 10 20 30 40 50" DEVICE=cuda:1 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
INIT=theta1 PORTIONS="60 70 80"             DEVICE=cuda:2 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
INIT=theta1 PORTIONS="100 90"               DEVICE=cuda:3 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
#   想更滿可調 MAX_PAR；想拆更細就再切 PORTIONS 丟更多卡（重疊會自動跳過）
```
**結果**（數字表 + 曲線圖）：
```bash
python3 thesis/chapter_4/aggregate_results.py --theta1   # θ¹ vs θ² 並排表 + 差值 → Table 4-2 θ¹ 欄
python3 thesis/chapter_4/plot_portion_curve.py           # 4.3 曲線：θ_rand/θ_ImageNet/θ¹/θ² 四線（θ¹ 隨 run 補滿）
```

---

## B. θ²_SimCLR bs×ep 熱力圖補完 — 4.3

目的：把各 ρ 的 bs×ep grid 填滿，驗證 bs↑/epoch↑ → 下游↑。
**30 個 SimCLR checkpoint 已存在 → 純下游 finetune，不需 pretrain。**

```bash
# (1) 補資料（缺口由 aggregate_results.py --heatmap 算出）
DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_heatmap_fill.sh rho100   # ρ=100：補 12 格（seed42）
DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_heatmap_fill.sh rho10    # ρ=10 ：補 5 格（bs128 列, 5 seeds）
DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_heatmap_fill.sh rho30    # ρ=30 ：整張 30 格（目前 0/30；SEEDS_30 預設 42）

# (2) 檢查覆蓋（文字 grid，標出 NA）
python3 thesis/chapter_4/aggregate_results.py --heatmap           # 現在含 ρ=100 / 30 / 10

# (3) 畫 PNG 熱力圖（每次都讀最新 JSON，天生 up-to-date；論文那張就是這支畫的）
cd classification/exp/weights_init
python3 plot_simclr_heatmap.py --portion 100     # 預設 5 seeds, aug4, simclr_lr0.0002, best-down-lr
python3 plot_simclr_heatmap.py --portion 100  --bar
python3 plot_simclr_heatmap.py --portion 30
python3 plot_simclr_heatmap.py --portion 10
#   → 存成 simclr_heatmap_aug4_portion{N}_simclrlr0p0002.png（可加 --save_dir 指定路徑）
```
**結果**：`--heatmap` = 文字覆蓋檢查；`plot_simclr_heatmap.py` = 實際 PNG 圖。
注意：ρ=100 為單 seed42；ρ=30 目前 0/30 全空、要先跑 rho30 才畫得出來。若要寫「bs 好處隨 portion 變大」的 interaction，需 10/30/100 對齊 seeds（見 CLAUDE.md）。

---

## C. θ²_SimCLR 中間 portion 補多 seed — 4.3（讓 Table 4-2 θ² 欄完整、紫線畫滿）

θ² best cfg（lr0.0002/bs256/ep500）目前多 seed 只在 ρ=2.5/10/100；中間 ρ 只有 seed42 → portion 曲線稀疏。
用**同一支 meta script**（INIT=theta2）補 ρ∈{5,20,30,40,50,60,70,80,90} × 5 seeds。

```bash
# 同一支 meta，只是 INIT=theta2、跑中間 portion（可分卡）
INIT=theta2 PORTIONS="5 20 30 40" DEVICE=cuda:4 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
INIT=theta2 PORTIONS="50 60 70"   DEVICE=cuda:5 ./thesis/chapter_4/run_4_3_simclr_finetune.sh
INIT=theta2 PORTIONS="100 80 90"      DEVICE=cuda:6 ./thesis/chapter_4/run_4_3_simclr_finetune.sh

python3 thesis/chapter_4/aggregate_results.py            # θ² 那欄補滿
python3 thesis/chapter_4/plot_portion_curve.py           # 紫線（θ²）用多 seed 畫滿
```

---

## D. 主動學習 — 4.4 ✅ 已可執行

協定（option A）：5 個 initial random seed（每 seed 一條軌跡，初始 pool 由 seed 隨機選）；
每個 portion **掃多個下游 lr**（`lr_grid_for`：ρ<20→`5e-5 1e-4 3e-4`，否則 `1e-4 3e-4 5e-4`），
**用 val loss 最低的 model 當選取器**去選下一批；std 取自 5 seeds；**ρ=2.5→60、interval 2.5**（24 點，停在 60）。
`random` 是合法策略（passive baseline，每步隨機選）。演算法用正確版（`diversity_correct`/`hybrid_correct`）。

```bash
# 跑單一策略 × 5 seeds（例：entropy）。可拆 seed 分卡加速（wrapper 內是序列跑）
# DEVICE=cuda:9 STRATEGIES="entropy" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

## 改一下LR search range

DEVICE=cuda:9 STRATEGIES="entropy" SEEDS="24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:4 STRATEGIES="margin" SEEDS="10 24" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下
DEVICE=cuda:4 STRATEGIES="margin" SEEDS="38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:3 STRATEGIES="conf" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下
DEVICE=cuda:6 STRATEGIES="conf" SEEDS="57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:1 STRATEGIES="coreset" SEEDS="42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:5 STRATEGIES="coreset" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:3 STRATEGIES="badge" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下
DEVICE=cuda:2 STRATEGIES="badge" SEEDS="57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:1 STRATEGIES="typiclust"      SEEDS="10 24" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:2 STRATEGIES="typiclust"      SEEDS="38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下


DEVICE=cuda:2 STRATEGIES="cluster_margin" SEEDS="10 24" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:1 STRATEGIES="cluster_margin" SEEDS="38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:5 STRATEGIES="cluster_margin" SEEDS="57" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

DEVICE=cuda:1 STRATEGIES="cluster_margin" SEEDS="42" ./thesis/chapter_4/run_4_4_active_learning.sh # 已下

python3 thesis/chapter_4/plot_al_curve.py
python3 thesis/chapter_4/al_curve_each_best.py
```

- **可用策略（8 個）**：`random conf entropy margin coreset badge` + 新增 **`typiclust`**（Diversity，低預算 SOTA，低 ρ 可能贏過現有）與 **`cluster_margin`**（Hybrid，比 BADGE 輕、效果相近的對照）。跑法相同：
  ```bash
  DEVICE=cuda:N STRATEGIES="typiclust"      SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh
  DEVICE=cuda:N STRATEGIES="cluster_margin" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh
  ```
  出處/公式/ε 選擇見 [`../CLAUDE.md`](../CLAUDE.md) §4.4。畫圖時 Diversity 多一條 TypiClust、Hybrid 多一條 Cluster-Margin（同色調不同細節色）。
- **每 portion 的 labeled id** 另存：`AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json`
  結構：`{portion: {lrs_swept, n_cumulative, selected, cumulative}}`（reproduce + Ch5 視覺化）。
- lr 用 **val（非 test）**挑最佳 → 避免 test leakage（`train_model` 多回傳 best_val_loss）。
- ⚠️ 舊 AL 資料已備份到 `AL_simclr_old/`（buggy coreset/badge + 舊 ckpt）；新跑會建乾淨的 `AL_simclr/`。

---

## F. 4.5 綜合比較（factor ablation）— Aug × Init × AL 各自貢獻

目的：4.2–4.4 是「逐一疊加」策略（AL 只在有 aug4 + θ² 的情況下做過）；4.5 拆解三策略**各自**的影響量。
兩個 Table（rows ρ = 10 / 30 / 50%；AL 代表策略 = **margin**，4.4 所有方法中最好、已定案 2026-06-12）：
- **Table 1（一次只開一個）**：Data Aug (4x) | Weight Init (θ²) | AL (margin) | All Three
- **Table 2（一次只關一個）**：w/o Data Aug | w/o Weight Init | w/o AL | All Three

「關」的定義：Aug 關 = no_aug；Init 關 = ImageNet；AL 關 = passive random（cold-start）。
7 個 cell 中 3 個直接用主實驗既有資料（aug_only=4.2、wo_al=4.3 θ²、all_three=4.4 margin），
只需新跑 4 個 arm，結果**隔離**在 `classification/exp_results/chapter4_5_ablation/{arm}/`（勿與主實驗混）。

```bash
# cold-start arm（最便宜：只跑 ρ=10/30/50 三點 × 5 seeds × 3 runs × lr 網格；MAX_PAR 並行）
ARM=init_only DEVICE=cuda:0 ./thesis/chapter_4/run_4_5_ablation.sh

# 三條 AL 軌跡（margin，ρ=2.5→50、interval 2.5、5 seeds；可用 SEEDS 拆卡並行）
ARM=al_only  DEVICE=cuda:1 ./thesis/chapter_4/run_4_5_ablation.sh   # no_aug + ImageNet + margin
ARM=wo_aug   DEVICE=cuda:2 ./thesis/chapter_4/run_4_5_ablation.sh   # no_aug + θ²       + margin
ARM=wo_init  DEVICE=cuda:3 ./thesis/chapter_4/run_4_5_ablation.sh   # aug4   + ImageNet + margin

# 檢視兩個 Table（缺 cell 標 NA；另印「三策略全關」參考 baseline）
python3 thesis/chapter_4/aggregate_4_5.py
```

注意事項：
- **lr/seed/runs 慣例與主實驗對齊**：cold-start arm 用 4.3 的 per-portion lr 網格 × 3 runs；
  AL arm 用 4.4 option A（sweep `3e-5 5e-5 1e-4 3e-4`、每 lr 1 run、best-val 選取器）；皆 5 seeds（10 24 38 42 57）。
- AL arm 的**初始 2.5% 一律 sweep**（獨立 exp_path 下查不到 cold-start best-lr 參照）——三個新 AL
  arm 內部一致；與 all_three（4.4 主跑，2.5% 用 cold-start best-lr）有此微小差異，論文不需提。
- **重跑安全**：AL arm 的 (strategy,seed) JSON 已存在就跳過（`FORCE=1` 強制接續）；
  cold-start arm 由 `check_existing_results`（3 runs 滿）自動跳過。
- 要換 AL 代表策略：三個 AL arm 加 `STRATEGY=xxx` 重跑即可（檔名含策略名、可並存；
  `aggregate_4_5.py --strategy xxx` 對應檢視；init_only 與策略無關不用重跑）。

---

## 圖表工具

> **共同慣例**：所有曲線圖都畫 **Target = 88.2%** 黑虛線（legend 寫 `Target`）。
> 4.2 與 4.3 兩張曲線在論文連續出現 → **配色刻意不重疊**：4.3 用 gray/green/orange/purple，4.2 改用 blue/red/teal/brown/pink，且 4.2 各線 marker 互異（o/s/^/D/v）。

### 4.2 資料增強曲線（θ_ImageNet：w/o / HF / VF / HF+VF / HF+VF+HVF × ρ）
```bash
cd classification/exp/data_aug
python3 plot_all.py                 # → 存 thesis/chapter_4/figs/imagenet_aug.png（順帶印 per-seed best-lr 統計）
#   --plot_rhos / --plot_xticks 可改 portion 範圍；--only_lr 固定下游 lr（預設 best-lr per run）
```
- 5 seeds（10/24/38/42/57）跨 seed pool，best-lr per ρ；ρ=100 用 seed42 檔內 mean±std。
- 配色/marker 見上方共同慣例（避開 4.3）。含 Target 線。輸出 `thesis/chapter_4/figs/imagenet_aug.png`（與其他 Ch4 圖同目錄）。

### 4.2 翻轉示意圖（4x aug 的 2×2 四宮格：原圖 / HF / VF / HVF）
挑一張 testing 影像，畫 (a) Original、(b) HF、(c) VF、(d) HVF 的 2×2 大圖（各子圖含 caption，碩論 Arial/dpi300 風格）。
```bash
python3 thesis/chapter_4/plot_flip_illustration.py \
  --image 'ds/classification/seven_class/train/Normal/20220225_153442B.png' \
  --gray --out thesis/chapter_4/figs/flip_illustration.png
#   --gray：OCT 灰階顯示（建議加）；--title：可選整體標題（一般留給 LaTeX caption）
```

### 4.3 portion 曲線（θ_rand / θ_ImageNet / θ¹ / θ² × ρ）
```bash
python3 thesis/chapter_4/plot_portion_curve.py
#   --simclr_lr/--simclr_bs/--simclr_ep 預設 0.0002/256/500（兩種 SimCLR 都用最大設定）
#   --all_portions 才畫 2.5-step 細點；預設只畫 canonical 2.5,5,10,20,...,100
```
- legend 用數學記號（θ_rand, θ_ImageNet, θ¹_SimCLR, θ²_SimCLR）；與 `aggregate_results.py` 同邏輯（best-lr per ρ、跨 seed pool），數字一致。
- 有資料才畫：θ¹ 隨 A3 補滿、θ² 中間 ρ 隨 C 補滿。輸出 `figs/portion_curve.png`。

### SimCLR 預訓練曲線（雙子圖：左 InfoNCE loss↓、右 top1/top5 acc↑）
```bash
# 單一 init
python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --init theta2 --bs 256 --ep 500
python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --init theta1 --bs 256 --ep 500 --lr 4e-4

# θ¹ vs θ² overlay（最大 bs/ep，default lr=2e-4；θ¹ 未跑完也會先畫現有 epoch）
python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --compare --bs 256 --ep 500

# 同一 init（預設 θ²）、固定 ep，疊不同 batch size（右圖只畫 Top-1）
python3 thesis/chapter_4/plot_simclr_pretrain_curve.py --sweep_bs --ep 500
#   --bs_list 預設 "16 32 64 128 256"；--init theta1 可改畫 θ¹
```
- `--init theta2`(ImageNet init)/`theta1`(random init)；`--bs`/`--ep` 必填；`--lr` 預設 0.0002（需與檔名一致）。
- `--compare`：θ¹ vs θ² overlay。左=loss 兩線；右 4 線（顏色=init、線型=Top-1 實/Top-5 虛，兩組 legend）。`--lr1`/`--lr2` 各指定 θ¹/θ² 的 lr（皆預設 0.0002）。θ¹ 未訓練完會印 partial 警告，跑完後重跑即正式圖。
- `--sweep_bs`：同一 init、固定 `--ep`，疊不同 batch size（viridis 由小到大）。左=loss、右=**只畫 Top-1**。輸出 `simclr_curve_bssweep_{init}_ep{ep}.png`。
- 無整體標題，只保留 (a)/(b) 子圖標題。論文 style（sans-serif、大字級、dpi 300）。
- 輸出 `thesis/chapter_4/figs/simclr_curve_{init}_bs{bs}_ep{ep}_lr{lr}.png`（compare 為 `..._compare_bs{bs}_ep{ep}.png`）。

---

## E. GradCAM 分析 — 4.2 資料增強的定性佐證

目的：用 Grad-CAM++ 看「4x aug 訓練的模型 vs w/o Aug」對病灶的注意力差異（驗證翻轉→上下左右更對稱地看病灶的 motivation）。
- **方法**（沿用學長 notebook `thesis/gradcam/Lesion classification.ipynb`）：Grad-CAM++，target layer = ResNet-18 的 **`layer4[-1]`**（plain resnet→`model.layer4[-1]`；SimCLR→`model.backbone.layer4[-1]`，腳本自動偵測）。
- 工具：`thesis/gradcam/gradcam_view.py`（自帶實作、免裝 pytorch_grad_cam；前處理用 repo eval transform）。
- ⚠️ 目前**沒存任何 fine-tuned ckpt** → 先用 `run_first_iter.py --save_ckpt`（非互動存檔；此模式會**跳過 check_existing、不寫結果 JSON**，純訓練+存 ckpt）訓 5 個 aug 條件。

### E1. 訓練 + 存 5 個 aug 條件的 ckpt（θ_ImageNet, ρ=100, seed42；各約幾分鐘，`--device` 挑空卡）
```bash
# 1x w/o Aug
python3 classification/run_first_iter.py --task_type hard --pretrained_weights imagenet \
  --portion 100 --seed 42 --no_data_aug --lr 1e-4 --epoch 20 --device cuda:8 \
  --save_ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth # 已跑完
# 2x HF（水平翻轉）
python3 classification/run_first_iter.py --task_type hard --pretrained_weights imagenet \
  --portion 100 --seed 42 --aug_factor 2 --flip_type horizontal --lr 1e-4 --epoch 20 --device cuda:8 \
  --save_ckpt thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth # 已跑完
# 2x VF（垂直翻轉） 
python3 classification/run_first_iter.py --task_type hard --pretrained_weights imagenet \
  --portion 100 --seed 42 --aug_factor 2 --flip_type vertical --lr 1e-4 --epoch 20 --device cuda:8 \
  --save_ckpt thesis/gradcam/ckpt/imagenet_p100_2x_vf.pth
# 3x HF+VF
python3 classification/run_first_iter.py --task_type hard --pretrained_weights imagenet \
  --portion 100 --seed 42 --aug_factor 3 --lr 1e-4 --epoch 20 --device cuda:0 \
  --save_ckpt thesis/gradcam/ckpt/imagenet_p100_3x.pth
# 4x HF+VF+HVF
python3 classification/run_first_iter.py --task_type hard --pretrained_weights imagenet \
  --portion 100 --seed 42 --aug_factor 4 --lr 1e-4 --epoch 20 --device cuda:8 \
  --save_ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth
```
aug 條件對應：`--no_data_aug`=1x；`--aug_factor 2 --flip_type horizontal/vertical`=HF/VF 2x；`--aug_factor 3`=3x；`--aug_factor 4`=4x。

### E2. 找 cherry-pick 候選（aug 對預測差別最大的影像）
對 val 7 類**全部影像**做完整 inference，算 `Δ = P_aug4(真實類別) − P_base(真實類別)`，**依類別印每個 label 的 top-5**——即「aug4 比 baseline 對正確類別提升最多」的影像，拿來看 GradCAM 對比最明顯。一次可給多個 baseline（no-aug、HF…），同一次 inference 就同時得到「aug4 vs no-aug」與「aug4 vs HF」兩份排序。
```bash
python3 thesis/gradcam/rank_aug_examples.py \
  --ckpt_aug4 thesis/gradcam/ckpt/imagenet_p100_4x.pth \
  --base noaug:thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth \
  --base HF:thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth \
  --device cuda:0 --topk 5 --out thesis/gradcam/out/rank_aug4.csv
#   每列還會標 pred:base→aug4（如 Normal→Eczema = baseline 判錯、aug4 改對）
#   --base 可重複給任意 baseline（如再加 VF:..._2x_vf.pth）；完整排序存進 --out 的 CSV
```
挑出名單後，把該 `--image` 餵給 E3 的 `gradcam_view.py` 看熱圖對比。

### E3. 畫 GradCAM（同一張影像比 w/o vs 4x）
```bash
## no aug
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_095624B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/eczema_20210421_095624B.png

#### 這張還蠻有代表性的!
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_100806B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/eczema_20210421_100806B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth  \
  --image ds/classification/seven_class/val/Vitiligo/20211210_160451B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/vitiligo_20211210_160451B.png

#### 這張也不錯!
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth  \
  --image 'ds/classification/seven_class/val/Seborrhoeic keratosis/20210305_100718B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/SK_20211210_160451B.png

#### 這張也很好! SL中，增厚的角質層
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_1x_noaug.pth  \
  --image 'ds/classification/seven_class/val/Solar lentigo/20220209_103428B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/SL_20220209_103428B.png



## HF
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_100806B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug2_hf_eczema_20210421_100806B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth  \
  --image 'ds/classification/seven_class/val/Seborrhoeic keratosis/20210305_100718B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug2_hf_SK_20211210_160451B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_2x_hf.pth  \
  --image 'ds/classification/seven_class/val/Solar lentigo/20220209_103428B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug2_hf_SL_20220209_103428B.png

## VF
python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_2x_vf.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_100806B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug2_vf_eczema_20210421_100806B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_2x_vf.pth  \
  --image 'ds/classification/seven_class/val/Seborrhoeic keratosis/20210305_100718B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug2_vf_SK_20211210_160451B.png


## 4 aug

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_095624B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug4_eczema_20210421_095624B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth  \
  --image ds/classification/seven_class/val/Eczema/20210421_100806B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug4_eczema_20210421_100806B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth  \
  --image ds/classification/seven_class/val/Vitiligo/20211210_160451B.png \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug4_vitiligo_20211210_160451B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth  \
  --image 'ds/classification/seven_class/val/Seborrhoeic keratosis/20210305_100718B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug4_SK_20211210_160451B.png

python3 thesis/gradcam/gradcam_view.py \
  --ckpt thesis/gradcam/ckpt/imagenet_p100_4x.pth  \
  --image 'ds/classification/seven_class/val/Solar lentigo/20220209_103428B.png' \
  --task_type hard --method gradcam++ --device cuda:0 \
  --out thesis/gradcam/out/aug4_SL_20220209_103428B.png
```
- 輸出三連圖（原圖 | 熱圖 | overlay）；`--target_class N` 指定對某類算 CAM（預設用預測類別）；`--method gradcam` 切回經典版。
- 之後若要強化：可加「flip-consistency / heatmap 對稱性」量化指標（全 test set），直接驗證「4x→注意力更對稱」的假設（見 `../CLAUDE.md` 想法）。

### E4. 一鍵：選圖 + 四 aug GradCAM + 綜合 2×2 圖（論文用對比圖）
`gradcam_panels.py` 自動串好整條：full inference（noaug/HF/VF/aug4）→ 每類別挑「aug4−noaug」與「aug4−HF」各 top-5（aug4 機率較高）→ 對每張選中圖畫四種 aug 的 GradCAM + 一張綜合 2×2 圖。
```bash
# (A) 自動選圖（每類別 aug4−noaug / aug4−HF 各 top-5）
python3 thesis/gradcam/gradcam_panels.py --device cuda:8
#   預設讀 thesis/gradcam/ckpt/imagenet_p100_{1x_noaug,2x_hf,2x_vf,4x}.pth；--topk 改每類張數

# (B) 指定影像（跳過排序，直接對你挑的代表性圖做 panel）
python3 thesis/gradcam/gradcam_panels.py --device cuda:8 --images \
  'ds/classification/seven_class/val/Solar lentigo/20220209_103428B.png' \
  'ds/classification/seven_class/val/Vitiligo/20211210_160451B.png'
#   類別取自影像上層資料夾名；結果一樣存到 out/<class>/<imgid>/
```
- 產出結構：`thesis/gradcam/out/<疾病>/<imgid>/`，內含 `panel.png`（綜合圖）+ `{noaug,HF,VF,aug4}_cam.png` + `original.png`；選圖清單 `out/panels_selected.csv`。
- 綜合 2×2：(a) Original（title=GT 簡寫，follow 碩論：Normal→Healthy、SK、SL）、(b) w/o Aug、(c) HF、(d) HF+VF+HVF，後三格 title 標 `P = x.xx`（模型對真實類別的 softmax 機率；GradCAM 也針對真實類別）。Arial/dpi300。

---

## 產出資料夾對照
| 實驗 | 結果路徑 |
|---|---|
| θ_rand / θ_ImageNet | `classification/exp_results/classification_hard/cold_start_{random,imagenet}/` |
| θ²_SimCLR（含熱力圖） | `.../cold_start_simclr/` |
| θ¹_SimCLR | `.../cold_start_simclr_randinit/`（新增，A 步驟產生）|
| 主動學習 | `.../AL_simclr/` |
| 4.5 factor ablation（F 步驟） | `classification/exp_results/chapter4_5_ablation/{init_only,al_only,wo_aug,wo_init}/` |
| SimCLR checkpoints | `SSL/simclr/ckpt/`（`.pkl`，不進 git）|
| GradCAM ckpt / 圖 | `thesis/gradcam/ckpt/`（`.pth`，不進 git）/ `thesis/gradcam/out/` |




```bash
DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_baselines.sh   # random + imagenet
DEVICE=cuda:8 ./thesis/chapter_4/run_4_3_learning_curve_simclr.sh      # θ¹ + θ²


DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s10.sh   # random + imagenet
DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_simclr_s10.sh      # θ¹ + θ²


DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s24.sh   # random + imagenet
DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_simclr_s24.sh      # θ¹ + θ²

DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s38.sh
DEVICE=cuda:7 ./thesis/chapter_4/run_4_3_learning_curve_simclr_s38.sh


DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_learning_curve_baselines_s57.sh
DEVICE=cuda:9 ./thesis/chapter_4/run_4_3_learning_curve_simclr_s57.sh


DEVICE=cuda:0 ./thesis/chapter_4/run_4_3_learning_curve_p100_runs.sh
```