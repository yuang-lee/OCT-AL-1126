import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 與 4.3 portion_curve 對齊的碩論樣式（字級、字體、線寬）
FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 18
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})


def _sstd(x):
    """樣本標準差（ddof=1，與 4.3/4.4 一致）；n<2 回 0。"""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_best_lr_stats(rho_data):
    """For a single rho's data (lr -> [vals]), find the best LR by mean accuracy."""
    best_mean = -1
    best_vals = []
    best_lr = None
    for lr, vals in rho_data.items():
        m = np.mean(vals)
        if m > best_mean:
            best_mean = m
            best_vals = vals
            best_lr = lr
    return np.mean(best_vals), np.std(best_vals), best_lr, best_vals


def get_best_lr_representative_acc(rho_data):
    """
    Find best LR by mean accuracy, return that mean as the representative acc
    for this JSON (seed) at this rho.
    """
    mean, std, best_lr, best_vals = get_best_lr_stats(rho_data)
    return mean, best_lr, best_vals


def get_fixed_lr_representative_acc(rho_data, only_lr):
    """
    Use the specified LR directly (if it exists), return that mean as the
    representative acc for this JSON (seed) at this rho.
    Compares as float to handle formats like '1e-4' vs '0.0001'.
    """
    target = float(only_lr)
    matched_key = None
    for k in rho_data:
        try:
            if float(k) == target:
                matched_key = k
                break
        except ValueError:
            continue
    if matched_key is None:
        return None, only_lr, []
    vals = rho_data[matched_key]
    return np.mean(vals), matched_key, vals


# 配色刻意避開 4.3 portion_curve 的 4 色（gray #7F7F7F / green #2CA02C /
# orange #E67E22 / purple #8E44AD），兩圖在碩論 4.2、4.3 連續出現不混淆。
# 每條線 marker 也不同，彼此好區隔。格式：(label, color, marker)
FIXED_CONFIGS = {
    "no_aug":          ("w/o Aug (1x)",     "#7F7F7F", "v"),   # gray（naive baseline 慣例：每圖都用灰）
    "aug2_horizontal": ("HF (2x)",          "#8C564B", "D"),   # brown
    "aug2_vertical":   ("VF (2x)",          "#17BECF", "^"),   # teal
    "aug3":            ("HF+VF (3x)",       "#D62728", "s"),   # red
    "aug4":            ("HF+VF+HVF (4x)",   "#1F77B4", "o"),   # blue（最佳，最醒目）
}

EXTRA_COLORS = [
    "#E74C3C", "#F39C12", "#1ABC9C", "#2980B9",
    "#8E44AD", "#D35400", "#16A085", "#C0392B",
]


def build_aug_configs(all_data_list):
    """Build aug configs from the union of all JSON files."""
    all_keys = set()
    for data in all_data_list:
        all_keys.update(data.keys())

    configs = {}
    for key, (label, color, marker) in FIXED_CONFIGS.items():
        if key in all_keys:
            configs[key] = (label, color, marker)
    extra_keys = [k for k in sorted(all_keys) if k not in FIXED_CONFIGS]
    for i, key in enumerate(extra_keys):
        color = EXTRA_COLORS[i % len(EXTRA_COLORS)]
        configs[key] = (key, color, "o")
    return configs


def get_all_rhos(all_data_list, aug_configs):
    """Get sorted union of all rho values across all JSONs and aug configs."""
    rhos = set()
    for data in all_data_list:
        for cfg in aug_configs:
            if cfg in data:
                for r in data[cfg]:
                    rhos.add(float(r))
    return sorted(rhos)


def get_representative_acc(rho_data, only_lr):
    """Dispatcher: use fixed LR or best LR depending on only_lr."""
    if only_lr is not None:
        return get_fixed_lr_representative_acc(rho_data, only_lr)
    else:
        return get_best_lr_representative_acc(rho_data)


def print_stats(all_data_list, json_paths, aug_configs, filter_rhos=None, only_lr=None, presented_keys=None):
    all_rhos = get_all_rhos(all_data_list, aug_configs)
    num_jsons = len(json_paths)

    if filter_rhos is not None:
        all_rhos = [r for r in all_rhos if r in filter_rhos]

    lr_mode_str = f"only_lr={only_lr}" if only_lr is not None else "best LR per run"

    # Print short names for each JSON
    print(f"\n{'='*90}")
    print(f"  JSON files ({num_jsons} total)  |  LR mode: {lr_mode_str}")
    for i, p in enumerate(json_paths):
        print(f"    [{i}] {p}")
    print(f"{'='*90}")

    for rho in all_rhos:
        rho_str = str(float(rho))
        rho_display = int(rho) if rho == int(rho) else rho

        print(f"\n{'='*90}")
        print(f"  rho = {rho_display}%")
        print(f"{'='*90}")

        for cfg_key, (label, _c, _m) in aug_configs.items():

            # Skip if not in presented_keys
            if presented_keys is not None and cfg_key not in presented_keys:
                continue

            # Collect per-JSON results
            per_json_results = []
            for i, data in enumerate(all_data_list):
                if cfg_key not in data or rho_str not in data[cfg_key]:
                    per_json_results.append(None)
                    continue
                mean, best_lr, best_vals = get_representative_acc(data[cfg_key][rho_str], only_lr)
                if mean is None:
                    per_json_results.append(None)
                    continue
                per_json_results.append({
                    'mean': mean,
                    'std': np.std(best_vals),
                    'best_lr': best_lr,
                    'best_vals': best_vals,
                })

            valid_results = [r for r in per_json_results if r is not None]
            if not valid_results:
                continue

            print(f"\n  [{label}]")
            print(f"  {'-'*96}")
            print(f"  {'JSON':<8} {'LR':<10} {'Mean':>8} {'Std':>8}  {'Values'}")
            print(f"  {'-'*96}")

            for i, result in enumerate(per_json_results):
                if result is None:
                    print(f"  [{i}]{'':<4} {'N/A':<10}")
                else:
                    vals_str = '[' + ', '.join(f'{v:.4f}' for v in result['best_vals']) + ']'
                    print(f"  [{i}]{'':<4} {result['best_lr']:<10} {result['mean']:>8.4f} {result['std']:>8.4f}  {vals_str}")

            # Cross-JSON summary
            representative_accs = [r['mean'] for r in valid_results]
            cross_mean = np.mean(representative_accs)
            cross_std = np.std(representative_accs)
            accs_str = '[' + ', '.join(f'{v:.4f}' for v in representative_accs) + ']'

            print(f"  {'-'*96}")
            print(f"  {'Cross-JSON':<18} {'Mean':>8} {'Std':>8}  Representative Accs")
            print(f"  {'':<18} {cross_mean:>8.4f} {cross_std:>8.4f}  {accs_str}")


SPECIAL_RHO_100_JSON = '../../exp_results/classification_hard/cold_start_imagenet/random42_bs16_ep20.json'


def plot(all_data_list, aug_configs, json_paths, save_path=None, only_lr=None, presented_keys=None, plot_rhos=None, plot_xticks=None):
    all_rhos = get_all_rhos(all_data_list, aug_configs)

    # Filter to only the requested rho values
    if plot_rhos is not None:
        plot_rhos_set = set(plot_rhos)
        all_rhos = [r for r in all_rhos if r in plot_rhos_set]

    fig, ax = plt.subplots(figsize=(12, 8))

    for cfg_key, (label, color, marker) in aug_configs.items():

        # Skip if not in presented_keys
        if presented_keys is not None and cfg_key not in presented_keys:
            continue
        rhos, means, stds = [], [], []

        for rho in all_rhos:
            rho_str = str(float(rho))

            # Special case: rho=100 → use only SPECIAL_RHO_100_JSON, inside-JSON mean & std
            if rho == 100.0:
                try:
                    special_idx = [os.path.abspath(p) for p in json_paths].index(
                        os.path.abspath(SPECIAL_RHO_100_JSON))
                    special_data = all_data_list[special_idx]
                except ValueError:
                    # fallback: match by basename
                    special_idx = next(
                        (i for i, p in enumerate(json_paths)
                         if os.path.basename(p) == os.path.basename(SPECIAL_RHO_100_JSON)), None)
                    special_data = all_data_list[special_idx] if special_idx is not None else None

                if special_data is None or cfg_key not in special_data or rho_str not in special_data[cfg_key]:
                    continue
                mean, _, vals = get_representative_acc(special_data[cfg_key][rho_str], only_lr)
                if mean is None:
                    continue
                rhos.append(rho / 100.0)
                means.append(mean)
                stds.append(_sstd(vals))
                continue

            # Normal case: cross-JSON mean & std
            representative_accs = []
            for data in all_data_list:
                if cfg_key not in data or rho_str not in data[cfg_key]:
                    continue
                mean, _, _ = get_representative_acc(data[cfg_key][rho_str], only_lr)
                if mean is None:
                    continue
                representative_accs.append(mean)

            if not representative_accs:
                continue

            rhos.append(rho / 100.0)
            means.append(np.mean(representative_accs))
            stds.append(_sstd(representative_accs))

        if not rhos:
            continue

        rhos  = np.array(rhos)
        means = np.array(means)
        stds  = np.array(stds)

        linestyle = '--' if cfg_key not in FIXED_CONFIGS else '-'

        ax.plot(rhos, means * 100, marker=marker, markersize=10, linewidth=3,
                label=label, color=color, linestyle=linestyle)
        ax.fill_between(rhos, (means - stds) * 100, (means + stds) * 100,
                        alpha=0.15, color=color)

    # Target 水平參考線（88.2%，黑虛線）
    ax.axhline(y=88.2, color='black', linestyle=(0, (8, 4)), linewidth=2.2,
               alpha=0.85, label='Target')

    ax.set_xlabel(r'Labeled Training Data Ratio $\rho$ (%)', fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_LABEL, labelpad=10)
    ax.tick_params(axis='both', labelsize=FONT_TICK, width=1.5, length=6)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x*100:.4g}')
    )
    xticks = plot_xticks if plot_xticks is not None else all_rhos
    ax.set_xticks([r / 100.0 for r in xticks])
    ax.set_ylim(45.5, 94.5)

    lr_title = f" (LR={only_lr})" if only_lr is not None else ""
    # 取得目前的 handles 和 labels
    handles, labels = ax.get_legend_handles_labels()

    # 定義你想要的順序（用 label 名稱）
    desired_order = [
        'HF+VF+HVF (4x)',
        'HF+VF (3x)',
        'VF (2x)',
        'HF (2x)',
        'w/o Aug (1x)',
        'Target',
    ]

    # 重新排序
    order = [labels.index(l) for l in desired_order if l in labels]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=FONT_LEGEND, framealpha=0.9, loc='lower right',
        title=f"LR mode: fixed{lr_title}" if only_lr else None
    )
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def print_summary_table(all_data_list, json_paths, aug_configs, presented_keys, plot_rhos, only_lr):
    """簡潔表：列 = 有畫的 portion，欄 = aug config，格子 = Cross-JSON mean±std(%)。
    = 每個 seed 取自己 best-lr 的 representative acc → 對 5 個 seed 取 mean/std（與曲線一致）；
    ρ=100 為 seed-independent，只用 seed42 檔內 runs 的 mean±std。"""
    keys = [k for k in aug_configs if (presented_keys is None or k in presented_keys)]
    labels = [aug_configs[k][0] for k in keys]
    CW = 14
    width = 8 + (CW + 1) * len(keys)
    # ρ=100 用的 seed42 檔索引
    sp_idx = next((i for i, p in enumerate(json_paths)
                   if os.path.basename(p) == os.path.basename(SPECIAL_RHO_100_JSON)), None)

    print("\n" + "=" * width)
    print(" 4.2 資料增強：各 ρ 的 mean±std(%)  [每 seed best-lr → over 5 seeds；ρ=100 為 seed42]")
    print("=" * width)
    print(f"{'ρ(%)':>6} | " + " ".join(f"{lb:>{CW}}" for lb in labels))
    print("-" * width)
    for rho in sorted(plot_rhos):
        r = str(float(rho))
        cells = []
        for k in keys:
            if rho == 100.0 and sp_idx is not None:
                d = all_data_list[sp_idx]
                if k in d and r in d[k]:
                    m, _, vals = get_representative_acc(d[k][r], only_lr)
                    cells.append(f"{m*100:5.2f}±{_sstd(vals)*100:4.2f}" if m is not None else "—")
                else:
                    cells.append("—")
            else:
                accs = []
                for d in all_data_list:
                    if k in d and r in d[k]:
                        m, _, _ = get_representative_acc(d[k][r], only_lr)
                        if m is not None:
                            accs.append(m)
                cells.append(f"{np.mean(accs)*100:5.2f}±{_sstd(accs)*100:4.2f}" if accs else "—")
        print(f"{rho:>6g} | " + " ".join(f"{c:>{CW}}" for c in cells))
    print("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_paths', type=str, nargs='+',
        default=[
            '../../exp_results/classification_hard/cold_start_imagenet/random10_bs16_ep20.json',
            '../../exp_results/classification_hard/cold_start_imagenet/random24_bs16_ep20.json',
            '../../exp_results/classification_hard/cold_start_imagenet/random38_bs16_ep20.json',
            '../../exp_results/classification_hard/cold_start_imagenet/random42_bs16_ep20.json',
            '../../exp_results/classification_hard/cold_start_imagenet/random57_bs16_ep20.json',
        ],
        help='List of JSON result files (one per seed/run)')
    parser.add_argument('--portions', type=float, nargs='*', default=None,
                        help='Only print stats for these portions (e.g. --portions 2.5 10 20). Default: print all.')
    parser.add_argument('--save', type=str,
                        default='../../../thesis/chapter_4/figs/imagenet_aug.png',
                        help='Path to save the plot. If not set, show interactively.')
    parser.add_argument('--only_lr', type=str, default=None,
                        help='If specified, only use this exact LR from all JSON files instead of picking the best LR per run.')
    parser.add_argument('--plot_rhos', type=float, nargs='+',
                        default=[2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='Which rho values to include in the plot. Default: [2.5, 5, 10, 20, 40, 60, 80, 100].')
    parser.add_argument('--plot_xticks', type=float, nargs='+',
                        default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='Which rho values to show as x-axis ticks. Default: same as --plot_rhos.')
    parser.add_argument('--presented_keys', type=str, nargs='+',
                        default=['no_aug', 'aug2_horizontal', 'aug2_vertical', 'aug3', 'aug4', 
                                #  'no_aug_jitter_b0.3_c0.3', 'aug4_jitter_b0.3_c0.3'
                                 ],
                        help='Which aug configs to show in terminal and plot. Default: the 5 base configs.')
    args = parser.parse_args()

    # If plot_xticks not specified, default to same as plot_rhos
    if args.plot_xticks is None:
        args.plot_xticks = args.plot_rhos

    all_data_list = []
    for p in args.json_paths:
        all_data_list.append(load_data(p))

    aug_configs = build_aug_configs(all_data_list)

    print(f"\nDetected aug configs: {list(aug_configs.keys())}")
    print(f"Presented keys: {args.presented_keys}")
    if args.only_lr is not None:
        print(f"LR mode: fixed LR = {args.only_lr}")
    else:
        print(f"LR mode: best LR per run (default)")

    print_summary_table(all_data_list, args.json_paths, aug_configs,
                        args.presented_keys, args.plot_rhos, args.only_lr)
    plot(all_data_list, aug_configs, json_paths=args.json_paths, save_path=args.save, only_lr=args.only_lr,
         presented_keys=args.presented_keys, plot_rhos=args.plot_rhos, plot_xticks=args.plot_xticks)


if __name__ == '__main__':
    main()