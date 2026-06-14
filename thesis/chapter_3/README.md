# Chapter 3 — 方法與資料集（圖表）

## Fig. 3-16　OCT 皮膚影像分類資料集類別分布

`plot_dataset_distribution.py`——**直接從磁碟數每類影像數**（不再手寫，避免出錯），
依數量遞減畫長條圖。輸出到 `figs/oct_class_distribution_{all,train}.png/.pdf`。

```bash
# 全資料集 (train+val 兩資料夾，2541)；論文 Fig 3-16 用這個
python3 thesis/chapter_3/plot_dataset_distribution.py --scope all
# AL training pool (只 train split，2032)
python3 thesis/chapter_3/plot_dataset_distribution.py --scope train
```

顯示名稱：Normal→**Healthy**、Solar lentigo→**SL**、Seborrhoeic keratosis→**SK**。

### 每類影像張數（all / train / val / test）

- 磁碟上只有 `train/` 與 `val/` 兩個資料夾。**all = train + val(disk) = 2541**。
- 論文的 **val / test 是把 `val/` 資料夾再 stratified 50/50 切開**
  （`classification/utils/data.py`：`train_test_split(test_size=0.5, stratify=targets, random_state=42)`），
  故 val=254、test=255（下表已重現該切分）。
- **AL 的 training pool = train split = 2032 張**（5.3 所有分析圖的 baseline 比例來源）。

| class (顯示名) | all | train | val | test |
|---|---:|---:|---:|---:|
| Healthy (Normal)            | 1011 | 808 | 101 | 102 |
| Nevus                       |  343 | 276 |  33 |  34 |
| SL (Solar lentigo)          |  338 | 270 |  34 |  34 |
| Eczema                      |  302 | 241 |  30 |  31 |
| Psoriasis                   |  225 | 180 |  23 |  22 |
| SK (Seborrhoeic keratosis)  |  172 | 137 |  18 |  17 |
| Vitiligo                    |  150 | 120 |  15 |  15 |
| **TOTAL**                   | **2541** | **2032** | **254** | **255** |

### ⚠️ 舊圖錯誤紀錄
舊版 `thesis/plot/class_dist.py` 的數字是**手動寫死**的，且填錯：
Nevus/SL 被低估、Eczema/Psoriasis 被高估、順序錯亂、總數只有 2441（少 100）。
本 script 改為從磁碟即時計數，根除此問題。
