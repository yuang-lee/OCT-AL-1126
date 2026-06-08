# 關於「不同 batch size 的 InfoNCE 不可直接比較」之文獻與 normalization

## 問題陳述（可寫進 caption / 內文）
本研究 SimCLR（`n_views=2`、batch size N）每個 anchor 面對 **M = 2N−1** 個候選（1 正 + 2N−2 負）。
因此 InfoNCE 是一個 (2N−1)-way 分類，其 **worst-case loss = log(2N−1)**、**chance Top-1 = 1/(2N−1)** 皆隨 batch size 變化。
→ 不同 batch size 的**原始 InfoNCE loss / contrastive accuracy 絕對值不可直接比較**；大 batch 的 loss 較高、contrastive top-1 收斂較慢，部分純粹來自任務難度（候選更多），非表徵較差。

## 各 normalization 手法 ↔ 可引用文獻

| 手法 | 數學 | 引用 |
|---|---|---|
| InfoNCE 是 MI 下界、含 log N 項 | `I(z_i;z_j) ≥ log M − L_InfoNCE` | **Oord et al. 2018 (CPC)**；**Poole et al. 2019** |
| 為何下界天花板 = log M、MI 大時退化（→ 即使 MI-normalize 也不完美） | 變分 MI 下界的 bias/variance 分析 | **Poole et al. 2019** |
| batch size 效益不只是「負樣本數」，而是梯度偏差 | gradient-bias 觀點 | **Chen et al. 2022 (NeurIPS)** |
| **與 batch size 無關**的表徵品質量度（建議正式比較用） | alignment、uniformity（hypersphere 幾何，不依賴 batch） | **Wang & Isola 2020** |
| SimCLR 本身（2(N−1) 負樣本、需大 batch） | — | **Chen et al. 2020 (SimCLR)** |

## 建議寫法（thesis framing）
1. `--sweep_bs` 曲線定位為 **training dynamics**，caption 註明絕對值不可比（cite Oord 2018 / Poole 2019）。
2. **「哪個 batch size 好」的排名用下游 linear-probe / fine-tune accuracy（熱力圖）**——這是 SSL 標準、與 batch size 無關的公平標尺（cite Wang & Isola 2020 對 downstream 的對應；SimCLR linear-eval 慣例 Chen 2020）。
3. 若要在 pretraining 空間給公平量化：報 **alignment & uniformity**（Wang & Isola 2020），或固定負樣本數 K 的 held-out InfoNCE。
4. 快速視覺修正（非必要）：畫 `log(2N−1) − L`（MI 下界估計，nats），但說明其天花板仍隨 N 變（Poole 2019）。

## 決定（2026-06-08）
- **不做 alignment/uniformity，也不寫其 eval code**（過於進階，且非必需）。
- **跨 batch size 的公平比較一律以「下游 finetune 熱力圖」為準**——這是 SSL 標準、與 pretraining batch size 任務難度無關的標尺，且資料已有。
- `--sweep_bs` 曲線僅作「training dynamics」配圖；若放，caption 註明「不同 batch size 的 InfoNCE 因候選數 2N−1 不同，絕對值不可直接比較」（cite Oord 2018 / Poole 2019）。留不留皆不影響論點。
- 只有當口試委員明確要求 pretraining 空間的量化比較，才回頭補 alignment/uniformity。

## BibTeX
```bibtex
@article{oord2018cpc,
  title={Representation Learning with Contrastive Predictive Coding},
  author={van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol},
  journal={arXiv preprint arXiv:1807.03748}, year={2018}}

@inproceedings{poole2019variational,
  title={On Variational Bounds of Mutual Information},
  author={Poole, Ben and Ozair, Sherjil and van den Oord, Aaron and Alemi, Alexander A. and Tucker, George},
  booktitle={ICML}, year={2019}}  % arXiv:1905.06922

@inproceedings{wang2020understanding,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={ICML}, pages={9929--9939}, year={2020}}  % arXiv:2005.10242

@inproceedings{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML}, year={2020}}  % arXiv:2002.05709

@inproceedings{chen2022why,
  title={Why Do We Need Large Batch Sizes in Contrastive Learning? A Gradient-Bias Perspective},
  author={Chen, Changyou and others}, booktitle={NeurIPS}, year={2022}}
```
（BibTeX 的作者/頁碼請於投稿前再核對官方頁面。）
