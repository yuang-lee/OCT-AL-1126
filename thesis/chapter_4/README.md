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
DEVICE=cuda:9 STRATEGIES="entropy" SEEDS="10 24 38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh
#   分兩卡：
DEVICE=cuda:1 STRATEGIES="entropy" SEEDS="10 24"    ./thesis/chapter_4/run_4_4_active_learning.sh
DEVICE=cuda:3 STRATEGIES="entropy" SEEDS="38 42 57" ./thesis/chapter_4/run_4_4_active_learning.sh

# 全部 6 策略 × 5 seeds（random+conf+entropy+margin+coreset+badge）
DEVICE=cuda:1 ./thesis/chapter_4/run_4_4_active_learning.sh

python3 thesis/chapter_4/plot_al_curve.py          # AL 曲線：Baseline(Random+100%) + 5 策略
```
- **每 portion 的 labeled id** 另存：`AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json`
  結構：`{portion: {lrs_swept, n_cumulative, selected, cumulative}}`（reproduce + Ch5 視覺化）。
- lr 用 **val（非 test）**挑最佳 → 避免 test leakage（`train_model` 多回傳 best_val_loss）。
- ⚠️ 舊 AL 資料已備份到 `AL_simclr_old/`（buggy coreset/badge + 舊 ckpt）；新跑會建乾淨的 `AL_simclr/`。

---

## 圖表工具

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

## 產出資料夾對照
| 實驗 | 結果路徑 |
|---|---|
| θ_rand / θ_ImageNet | `classification/exp_results/classification_hard/cold_start_{random,imagenet}/` |
| θ²_SimCLR（含熱力圖） | `.../cold_start_simclr/` |
| θ¹_SimCLR | `.../cold_start_simclr_randinit/`（新增，A 步驟產生）|
| 主動學習 | `.../AL_simclr/` |
| SimCLR checkpoints | `SSL/simclr/ckpt/`（`.pkl`，不進 git）|
