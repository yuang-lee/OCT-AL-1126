#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 4 結果彙整 / 對比工具。

從 classification/exp_results/ 撈出真實數字，用「best-lr per portion」慣例
（每個 ρ 掃過的多個 lr 中選 mean 最高者）計算 mean±std，並：
  - 印出 Table 4-2（θ_rand / θ_ImageNet / θ²_SimCLR）的 codebase 數字
  - 與論文 PDF 已填數字對比、標出 Δ
  - 印出主動學習 (4.4) 各策略曲線
  - 標出覆蓋缺口（哪些 ρ / seed 還沒跑）

用法（從 repo 根或任何地方都可，路徑用 script 位置推算）：
    python3 thesis/chapter_4/aggregate_results.py
    python3 thesis/chapter_4/aggregate_results.py --aug aug4
"""
import os, re, json, argparse
import numpy as np

# repo 根 = 本檔的上上層 (thesis/chapter_4/ -> repo root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP  = os.path.join(ROOT, "classification", "exp_results", "classification_hard")

PORTIONS = [2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]

# 論文 PDF (0427) 已填數字，供對比。None = 論文留空。
THESIS = {
    "theta_rand":    {2.5:41.28, 5:44.20, 10:48.47, 20:57.20, 30:66.67, 40:71.11,
                      50:75.61, 60:79.37, 70:82.14, 80:82.54, 90:85.15, 100:85.49},
    "theta_imagenet":{2.5:50.51, 5:58.59, 10:66.01, 20:74.85, 30:79.71, 40:82.43,
                      50:85.05, 60:87.63, 70:89.49, 80:90.80, 90:91.25, 100:92.29},
    "theta2_simclr": {2.5:55.90, 10:71.03, 100:96.17},   # 其餘論文留空
}


def _acc_list(vals):
    """JSON value 可能是 list（cold-start）或 {'acc': list, ...}（AL）。"""
    return vals["acc"] if isinstance(vals, dict) else vals


def best_lr_per_portion(portion_to_lrdict):
    """{ρ: {lr: [runs]}} -> {ρ: (mean%, std%, lr, n_runs)}，每 ρ 選 mean 最高的 lr。"""
    out = {}
    for p, lrd in portion_to_lrdict.items():
        bm, bs, bl, bn = -1.0, 0.0, None, 0
        for lr, vals in lrd.items():
            a = np.asarray(_acc_list(vals), dtype=float)
            if a.size == 0:
                continue
            m = a.mean()
            if m > bm:
                bm, bs, bl, bn = m, (a.std(ddof=1) if a.size > 1 else 0.0), lr, a.size
        out[float(p)] = (bm * 100, bs * 100, bl, bn)
    return out


def pool_seed_files(folder, match, aug="aug4"):
    """把資料夾中所有符合 match(filename) 的 seed 檔，於同 (ρ,lr) pool 起所有 runs。"""
    pooled = {}
    if not os.path.isdir(folder):
        return {}
    for f in os.listdir(folder):
        if not (f.endswith(".json") and match(f)):
            continue
        try:
            d = json.load(open(os.path.join(folder, f)))
        except Exception:
            continue
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            for lr, vals in lrd.items():
                pooled.setdefault(float(p), {}).setdefault(lr, []).extend(_acc_list(vals))
    return best_lr_per_portion(pooled)


def print_init_table(aug):
    print("=" * 78)
    print(f"Table 4-2  不同初始化 θ₀ × ρ 的分類準確率 (%)   [aug={aug}, best-lr per ρ]")
    print("=" * 78)

    cs = lambda sub: os.path.join(EXP, sub)
    rand = pool_seed_files(cs("cold_start_random"),
                           lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), aug)
    img  = pool_seed_files(cs("cold_start_imagenet"),
                           lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), aug)
    # θ²: best cfg 跨 seed（多 seed，僅少數 ρ）
    t2   = pool_seed_files(cs("cold_start_simclr"),
                           lambda f: "simclr_lr0.0002_simclr_bs256_simclr_ep500" in f, aug)
    # θ² 曲線（seed42 彙整檔，全 ρ 但單 seed）
    t2curve_path = cs("cold_start_simclr") + "/random42_bs16.json"
    t2curve = {}
    if os.path.isfile(t2curve_path):
        d = json.load(open(t2curve_path))
        if aug in d:
            t2curve = best_lr_per_portion(d[aug])

    cols = [("θ_rand", rand, THESIS["theta_rand"]),
            ("θ_ImageNet", img, THESIS["theta_imagenet"]),
            ("θ²_SimCLR(best cfg, multi-seed)", t2, THESIS["theta2_simclr"])]

    for name, data, th in cols:
        print(f"\n--- {name} ---")
        print(f"{'ρ%':>6} {'codebase':>15} {'lr':>8} {'n':>4} {'thesis':>8} {'Δ':>7}")
        for p in PORTIONS:
            if p in data and data[p][2] is not None:
                m, s, lr, n = data[p]
                t = th.get(p)
                ts = f"{t:.2f}" if t is not None else "  -"
                dd = f"{m-t:+.2f}" if t is not None else "   -"
                print(f"{p:>6} {m:>8.2f}±{s:<5.2f} {lr:>8} {n:>4} {ts:>8} {dd:>7}")
            else:
                miss = "(thesis有)" if th.get(p) is not None else ""
                print(f"{p:>6} {'(no data)':>15} {'':>8} {'':>4} {'':>8} {miss:>7}")

    print("\n--- θ²_SimCLR 曲線（seed42 彙整檔，圖的來源，單 seed 無跨 seed std）---")
    if t2curve:
        for p in sorted(t2curve):
            m, s, lr, n = t2curve[p]
            print(f"  ρ={p:>5}: {m:6.2f}±{s:<5.2f} (lr={lr}, n={n})")
    else:
        print("  (找不到 random42_bs16.json)")

    print("\n--- θ¹_SimCLR (random→SimCLR) ---")
    print("  ❌ 無任何資料：需先 `run.py -a resnet18_random` 預訓練，再 finetune。")


def print_al_table(aug):
    print("\n" + "=" * 78)
    print(f"4.4 主動學習 各策略準確率 (%)   [aug={aug}, best-lr per ρ]")
    print("=" * 78)
    al = os.path.join(EXP, "AL_simclr")
    if not os.path.isdir(al):
        print("  (找不到 AL_simclr/)"); return
    # 每策略跨所有 seed 檔 pool（同 portion,lr pool 全 runs），best-lr per portion
    pooled = {}   # strategy -> {portion: {lr: [runs]}}
    seeds = {}    # strategy -> set(seed)
    for f in sorted(os.listdir(al)):
        if not f.endswith(".json") or "copy" in f:
            continue
        strat = f.split("_")[0]
        m = re.search(r"seed(\d+)", f)
        d = json.load(open(os.path.join(al, f)))
        if aug not in d:
            continue
        seeds.setdefault(strat, set())
        if m:
            seeds[strat].add(m.group(1))
        for p, lrd in d[aug].items():
            for lr, vals in lrd.items():
                pooled.setdefault(strat, {}).setdefault(float(p), {}).setdefault(lr, []).extend(_acc_list(vals))
    if not pooled:
        print("  (無 aug4 資料)"); return
    strategies = {s: best_lr_per_portion(pd) for s, pd in pooled.items()}

    ps = sorted({p for s in strategies.values() for p in s})
    names = list(strategies)
    print("  策略(seeds): " + ", ".join(f"{n}({len(seeds[n])})" for n in names))
    print(f"{'ρ%':>6} " + " ".join(f"{n:>10}" for n in names))
    for p in ps:
        row = f"{p:>6} "
        for n in names:
            row += f"{strategies[n][p][0]:>10.2f} " if p in strategies[n] else f"{'-':>10} "
        print(row)
    print("  (值=best-lr per portion 的 mean，跨 seed pool；seed 數見上)")


def print_heatmap(aug, simclr_lr="0.0002"):
    """θ²_SimCLR 的 bs×ep 下游熱力圖：對 ρ=100 與 ρ=10 印出 best-lr mean 與覆蓋缺口。"""
    BS = [16, 32, 64, 128, 256]
    EP = [10, 20, 50, 100, 200, 500]
    folder = os.path.join(EXP, "cold_start_simclr")
    pat = re.compile(
        r"random(\d+)_bs16_ep20_simclr_lr" + re.escape(simclr_lr) +
        r"_simclr_bs(\d+)_simclr_ep(\d+)\.json")
    for rho in (100.0, 30.0, 10.0):
        # cell -> pooled downstream runs across seeds (best-lr)
        cell = {}
        for f in os.listdir(folder) if os.path.isdir(folder) else []:
            m = pat.match(f)
            if not m:
                continue
            bs, ep = int(m.group(2)), int(m.group(3))
            d = json.load(open(os.path.join(folder, f)))
            if aug in d and str(rho) in d[aug]:
                for lr, vals in d[aug][str(rho)].items():
                    cell.setdefault((bs, ep), {}).setdefault(lr, []).extend(_acc_list(vals))
        print("\n" + "=" * 78)
        print(f"θ²_SimCLR bs×ep heatmap @ ρ={rho:g}%  (simclr_lr={simclr_lr}, best-lr per cell)")
        print("=" * 78)
        print("bs\\ep  " + " ".join(f"{e:>7}" for e in EP))
        missing = []
        for bs in BS:
            row = f"{bs:>5}  "
            for ep in EP:
                if (bs, ep) in cell:
                    best = max((np.mean(v) for v in cell[(bs, ep)].values()))
                    row += f"{best*100:>7.2f}"
                else:
                    row += f"{'NA':>7}"; missing.append((bs, ep))
            print(row)
        print(f"  缺 {len(missing)}/30: {missing}" if missing else "  ✅ 全格已填")


def print_theta1(aug):
    """θ¹_SimCLR (random→SimCLR)：依 simclr_lr 分組印各 portion best-lr acc，
    並與 θ² (imagenet→SimCLR, best cfg) 並排，作為 θ²>θ¹ 的對照。"""
    folder = os.path.join(EXP, "cold_start_simclr_randinit")
    print("\n" + "=" * 78)
    print(f"θ¹_SimCLR (random→SimCLR) 結果  [aug={aug}, best-lr per ρ]")
    print("=" * 78)
    if not os.path.isdir(folder):
        print("  θ¹ 尚無資料。請先跑 run_4_3_theta1_pretrain.sh → _pick_lr.sh → _finetune.sh")
        return
    pat = re.compile(
        r"random(\d+)_bs16_ep20_simclr_lr([\d.e-]+)_simclr_bs(\d+)_simclr_ep(\d+)\.json")
    by_slr = {}   # simclr_lr -> {ρ: {down_lr: [runs]}}
    for f in os.listdir(folder):
        m = pat.match(f)
        if not m:
            continue
        slr = m.group(2)
        d = json.load(open(os.path.join(folder, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            for lr, vals in lrd.items():
                by_slr.setdefault(slr, {}).setdefault(float(p), {}).setdefault(lr, []).extend(_acc_list(vals))
    if not by_slr:
        print("  資料夾存在但無 aug4 資料。")
        return
    # 各 simclr_lr 在 ρ=10 與 100 的下游 acc（lr-pick 用；ρ=10 較能分辨預訓練好壞）
    print("\n--- pick-lr：各候選 simclr_lr 的下游 acc（best down-lr）---")
    print(f"  {'simclr_lr':>10} {'ρ=10%':>16} {'ρ=100%':>16}")
    best_slr, best_at10 = None, -1.0
    for slr in sorted(by_slr, key=float):
        st = best_lr_per_portion(by_slr[slr])
        c10 = f"{st[10.0][0]:.2f}±{st[10.0][1]:.2f}" if 10.0 in st else "-"
        c100 = f"{st[100.0][0]:.2f}±{st[100.0][1]:.2f}" if 100.0 in st else "-"
        print(f"  {slr:>10} {c10:>16} {c100:>16}")
        if 10.0 in st and st[10.0][0] > best_at10:
            best_at10, best_slr = st[10.0][0], slr
    if best_slr:
        print(f"  → 建議 θ¹ pretraining lr = {best_slr}（以 ρ=10% 為準={best_at10:.2f}；低 portion 最能反映預訓練品質）")
    else:
        print("  (ρ=10% 尚無資料；先跑 run_4_3_theta1_pick_lr.sh)")
    # headline：θ¹(best slr) vs θ²(best cfg) 跨 portion
    slr = best_slr or sorted(by_slr, key=float)[0]
    t1 = best_lr_per_portion(by_slr[slr])
    t2 = pool_seed_files(os.path.join(EXP, "cold_start_simclr"),
                         lambda f: "simclr_lr0.0002_simclr_bs256_simclr_ep500" in f, aug)
    print(f"\n--- θ¹ (random→SimCLR, lr={slr}) vs θ² (imagenet→SimCLR, lr2e-4/bs256/ep500) ---")
    print(f"{'ρ%':>6} {'θ¹':>16} {'θ²':>16} {'θ²-θ¹':>8}")
    for p in PORTIONS:
        a = f"{t1[p][0]:.2f}±{t1[p][1]:.2f}" if p in t1 else "-"
        b = f"{t2[p][0]:.2f}±{t2[p][1]:.2f}" if p in t2 else "-"
        d = f"{t2[p][0]-t1[p][0]:+.2f}" if (p in t1 and p in t2) else ""
        print(f"{p:>6} {a:>16} {b:>16} {d:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--heatmap", action="store_true", help="只印 bs×ep 熱力圖覆蓋")
    ap.add_argument("--theta1", action="store_true", help="只印 θ¹ (random→SimCLR) 結果與 vs θ² 對照")
    args = ap.parse_args()
    print(f"\nREPO = {ROOT}")
    if args.heatmap:
        print_heatmap(args.aug)
        return
    if args.theta1:
        print_theta1(args.aug)
        return
    print_init_table(args.aug)
    print_al_table(args.aug)
    print_heatmap(args.aug)
    print()


if __name__ == "__main__":
    main()
