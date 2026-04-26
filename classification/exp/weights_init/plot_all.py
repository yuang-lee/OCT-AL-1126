import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# Configs
# =============================================================================

BASE_INIT_CONFIGS = {
    "random": ("Random Init", "#888888"),
    "imagenet": ("ImageNet", "#27AE60"),
}

SIMCLR_COLOR = "#9B59B6"


# Expected filename:
# random42_bs16_ep20_simclr_lr0.0002_simclr_bs256_simclr_ep500.json
SIMCLR_JSON_PATTERN = re.compile(
    r"^(?P<seed>random\d+)"
    r"_bs(?P<downstream_bs>[^_]+)"
    r"_ep(?P<downstream_ep>[^_]+)"
    r"_simclr_lr(?P<simclr_lr>[^_]+)"
    r"_simclr_bs(?P<simclr_bs>[^_]+)"
    r"_simclr_ep(?P<simclr_ep>[^.]+)"
    r"\.json$"
)

STANDARD_SUFFIX_PATTERN = re.compile(
    r"^bs(?P<downstream_bs>[^_]+)_ep(?P<downstream_ep>[^.]+)(?:\.json)?$"
)


# =============================================================================
# Filename parsing
# =============================================================================

def parse_standard_suffix(suffix):
    """
    Parse downstream suffix, e.g.
    bs16_ep20.json -> downstream_bs=16, downstream_ep=20
    """
    m = STANDARD_SUFFIX_PATTERN.match(suffix)
    if not m:
        raise ValueError(
            f"Cannot parse suffix='{suffix}'. Expected format like 'bs16_ep20.json'."
        )

    return {
        "downstream_bs": m.group("downstream_bs"),
        "downstream_ep": m.group("downstream_ep"),
    }


def parse_simclr_json_filename(filename):
    m = SIMCLR_JSON_PATTERN.match(filename)
    if not m:
        return None

    item = m.groupdict()
    item["filename"] = filename
    return item


def safe_float_equal(a, b, atol=1e-12):
    try:
        return abs(float(a) - float(b)) <= atol
    except Exception:
        return str(a) == str(b)


def make_simclr_init_key(simclr_lr, simclr_bs, simclr_ep):
    return f"simclr__lr{simclr_lr}__bs{simclr_bs}__ep{simclr_ep}"


def make_simclr_label(simclr_lr, simclr_bs, simclr_ep):
    return f"SimCLR lr={simclr_lr}, bs={simclr_bs}, ep={simclr_ep}"


# =============================================================================
# SimCLR scanning
# =============================================================================

def scan_simclr_jsons(base_dir):
    """
    Scan cold_start_simclr/*.json.

    Returns
    -------
    records : list of dict
    """
    simclr_dir = os.path.join(base_dir, "cold_start_simclr")
    records = []
    unparsed = []

    print("")
    print("=" * 140)
    print("Scanning cold_start_simclr JSON files")
    print("=" * 140)
    print(f"Directory: {simclr_dir}")

    if not os.path.isdir(simclr_dir):
        print(f"Directory not found: {simclr_dir}")
        return records

    json_files = sorted(
        f for f in os.listdir(simclr_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(simclr_dir, f))
    )

    if not json_files:
        print("No JSON files found.")
        return records

    for filename in json_files:
        parsed = parse_simclr_json_filename(filename)

        if parsed is None:
            unparsed.append(filename)
            continue

        parsed["path"] = os.path.join(simclr_dir, filename)
        records.append(parsed)

    records = sorted(
        records,
        key=lambda x: (
            float(x["simclr_lr"]),
            int(float(x["simclr_bs"])),
            int(float(x["simclr_ep"])),
            x["seed"],
            int(float(x["downstream_bs"])),
            int(float(x["downstream_ep"])),
        ),
    )

    print("")
    print("=" * 140)
    print("All parsed SimCLR JSON files")
    print("=" * 140)

    if records:
        print(
            f"{'Seed':<10} "
            f"{'DownBS':>8} "
            f"{'DownEP':>8} "
            f"{'SimCLR_LR':>12} "
            f"{'SimCLR_BS':>10} "
            f"{'SimCLR_EP':>10}  "
            f"Filename"
        )
        print("-" * 140)

        for r in records:
            print(
                f"{r['seed']:<10} "
                f"{r['downstream_bs']:>8} "
                f"{r['downstream_ep']:>8} "
                f"{r['simclr_lr']:>12} "
                f"{r['simclr_bs']:>10} "
                f"{r['simclr_ep']:>10}  "
                f"{r['filename']}"
            )

        print("-" * 140)
        print(f"Parsed file count: {len(records)}")
    else:
        print("No parseable SimCLR JSON files found.")

    unique_settings = get_unique_simclr_settings(records)

    print("")
    print("=" * 140)
    print("All unique parsed SimCLR settings")
    print("=" * 140)

    if unique_settings:
        print(
            f"{'Index':>5} "
            f"{'SimCLR_LR':>12} "
            f"{'SimCLR_BS':>10} "
            f"{'SimCLR_EP':>10} "
            f"{'N_files':>8} "
            f"{'Seeds':<30} "
            f"{'Downstream(bs,ep)':<30}"
        )
        print("-" * 140)

        for i, s in enumerate(unique_settings, start=1):
            seeds_str = ", ".join(sorted(s["seeds"]))

            downstream_str = ", ".join(
                f"bs{bs}_ep{ep}"
                for bs, ep in sorted(
                    s["downstream_settings"],
                    key=lambda x: (int(float(x[0])), int(float(x[1]))),
                )
            )

            print(
                f"{i:>5} "
                f"{s['simclr_lr']:>12} "
                f"{s['simclr_bs']:>10} "
                f"{s['simclr_ep']:>10} "
                f"{s['file_count']:>8} "
                f"{seeds_str:<30} "
                f"{downstream_str:<30}"
            )

        print("-" * 140)
        print(f"Unique SimCLR setting count: {len(unique_settings)}")
    else:
        print("No unique SimCLR settings parsed.")

    if unparsed:
        print("")
        print("=" * 140)
        print("Unparsed JSON files under cold_start_simclr")
        print("=" * 140)
        for filename in unparsed:
            print(f"  {filename}")

    return records


def get_unique_simclr_settings(simclr_records):
    """
    Return unique SimCLR settings.

    Each output item contains:
    simclr_lr, simclr_bs, simclr_ep, seeds, downstream_settings, file_count
    """
    unique = {}

    for r in simclr_records:
        key = (
            str(r["simclr_lr"]),
            int(float(r["simclr_bs"])),
            int(float(r["simclr_ep"])),
        )

        if key not in unique:
            unique[key] = {
                "simclr_lr": str(r["simclr_lr"]),
                "simclr_bs": int(float(r["simclr_bs"])),
                "simclr_ep": int(float(r["simclr_ep"])),
                "seeds": set(),
                "downstream_settings": set(),
                "file_count": 0,
            }

        unique[key]["seeds"].add(r["seed"])
        unique[key]["downstream_settings"].add(
            (
                str(r["downstream_bs"]),
                str(r["downstream_ep"]),
            )
        )
        unique[key]["file_count"] += 1

    return sorted(
        unique.values(),
        key=lambda x: (
            float(x["simclr_lr"]),
            int(x["simclr_bs"]),
            int(x["simclr_ep"]),
        ),
    )


# =============================================================================
# Path helpers
# =============================================================================

def get_standard_json_paths(base_dir, init_key, seeds, suffix):
    return [
        os.path.join(base_dir, f"cold_start_{init_key}", f"{seed}_{suffix}")
        for seed in seeds
    ]


def find_simclr_record(records, seed, suffix, simclr_lr, simclr_bs, simclr_ep):
    suffix_info = parse_standard_suffix(suffix)

    downstream_bs = suffix_info["downstream_bs"]
    downstream_ep = suffix_info["downstream_ep"]

    for r in records:
        if r["seed"] != seed:
            continue
        if str(r["downstream_bs"]) != str(downstream_bs):
            continue
        if str(r["downstream_ep"]) != str(downstream_ep):
            continue
        if not safe_float_equal(r["simclr_lr"], simclr_lr):
            continue
        if int(float(r["simclr_bs"])) != int(float(simclr_bs)):
            continue
        if int(float(r["simclr_ep"])) != int(float(simclr_ep)):
            continue

        return r

    return None


def build_simclr_json_path(base_dir, seed, suffix, simclr_lr, simclr_bs, simclr_ep):
    suffix_no_json = suffix[:-5] if suffix.endswith(".json") else suffix

    filename = (
        f"{seed}_{suffix_no_json}"
        f"_simclr_lr{simclr_lr}"
        f"_simclr_bs{simclr_bs}"
        f"_simclr_ep{simclr_ep}"
        f".json"
    )

    return os.path.join(base_dir, "cold_start_simclr", filename)


def get_simclr_json_paths(
    base_dir,
    seeds,
    suffix,
    simclr_lr,
    simclr_bs,
    simclr_ep,
    simclr_records,
):
    paths = []

    for seed in seeds:
        record = find_simclr_record(
            records=simclr_records,
            seed=seed,
            suffix=suffix,
            simclr_lr=simclr_lr,
            simclr_bs=simclr_bs,
            simclr_ep=simclr_ep,
        )

        if record is not None:
            paths.append(record["path"])
        else:
            paths.append(
                build_simclr_json_path(
                    base_dir=base_dir,
                    seed=seed,
                    suffix=suffix,
                    simclr_lr=simclr_lr,
                    simclr_bs=simclr_bs,
                    simclr_ep=simclr_ep,
                )
            )

    return paths


# =============================================================================
# JSON loading
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data_list(paths, seeds, title):
    """
    Load one data list aligned to seeds.
    Missing files are represented by None.
    """
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)

    data_list = []

    for seed, path in zip(seeds, paths):
        try:
            data_list.append(load_json(path))
            print(f"  Loaded [{seed}]: {path}")
        except FileNotFoundError:
            data_list.append(None)
            print(f"  N/A [{seed}]: {path}")
        except json.JSONDecodeError as e:
            data_list.append(None)
            print(f"  JSON decode error [{seed}]: {path}")
            print(f"    {e}")

    return data_list


# =============================================================================
# LR selection
# =============================================================================

def get_best_lr_representative_acc(rho_data):
    """
    Pick the downstream classifier LR with the highest mean accuracy.
    """
    best_mean = None
    best_vals = None
    best_lr = None

    for lr, vals in rho_data.items():
        if vals is None or len(vals) == 0:
            continue

        mean_val = float(np.mean(vals))

        if best_mean is None or mean_val > best_mean:
            best_mean = mean_val
            best_vals = vals
            best_lr = lr

    if best_mean is None:
        return None, None, []

    return float(np.mean(best_vals)), best_lr, best_vals


def get_fixed_lr_representative_acc(rho_data, only_lr):
    """
    Use a fixed downstream classifier LR.
    """
    target = float(only_lr)
    matched_key = None

    for key in rho_data.keys():
        try:
            if float(key) == target:
                matched_key = key
                break
        except ValueError:
            continue

    if matched_key is None:
        return None, only_lr, []

    vals = rho_data[matched_key]

    if vals is None or len(vals) == 0:
        return None, matched_key, []

    return float(np.mean(vals)), matched_key, vals


def get_representative_acc(rho_data, only_lr):
    if only_lr is not None:
        return get_fixed_lr_representative_acc(rho_data, only_lr)

    return get_best_lr_representative_acc(rho_data)


# =============================================================================
# Stats helpers
# =============================================================================

def get_all_rhos(init_data_dict, aug_key):
    rhos = set()

    for data_list in init_data_dict.values():
        for data in data_list:
            if data is None:
                continue

            if aug_key not in data:
                continue

            for rho in data[aug_key].keys():
                rhos.add(float(rho))

    return sorted(rhos)


def print_stats(
    init_data_dict,
    init_config_dict,
    presented_keys,
    seeds,
    aug_key,
    filter_rhos=None,
    only_lr=None,
):
    all_rhos = get_all_rhos(init_data_dict, aug_key)

    if filter_rhos is not None:
        filter_rhos = set(float(x) for x in filter_rhos)
        all_rhos = [rho for rho in all_rhos if rho in filter_rhos]

    lr_mode_str = (
        f"fixed downstream LR={only_lr}"
        if only_lr is not None
        else "best downstream LR per JSON"
    )

    print("")
    print("=" * 100)
    print(f"  Aug key: {aug_key}  |  LR mode: {lr_mode_str}")
    print(f"  Seeds: {seeds}")
    print("=" * 100)

    if not all_rhos:
        print("No rho values found for the requested configuration.")
        return

    for rho in all_rhos:
        rho_str = str(float(rho))
        rho_display = int(rho) if rho == int(rho) else rho

        print("")
        print("=" * 100)
        print(f"  rho = {rho_display}%")
        print("=" * 100)

        for init_key in presented_keys:
            if init_key not in init_data_dict:
                continue
            if init_key not in init_config_dict:
                continue

            label, _ = init_config_dict[init_key]
            data_list = init_data_dict[init_key]

            per_seed_results = []

            for data in data_list:
                if data is None:
                    per_seed_results.append(None)
                    continue

                if aug_key not in data:
                    per_seed_results.append(None)
                    continue

                if rho_str not in data[aug_key]:
                    per_seed_results.append(None)
                    continue

                mean_val, selected_lr, vals = get_representative_acc(
                    data[aug_key][rho_str],
                    only_lr=only_lr,
                )

                if mean_val is None:
                    per_seed_results.append(None)
                    continue

                per_seed_results.append(
                    {
                        "mean": mean_val,
                        "std": float(np.std(vals)),
                        "lr": selected_lr,
                        "vals": vals,
                    }
                )

            valid = [x for x in per_seed_results if x is not None]

            if not valid:
                continue

            print("")
            print(f"  [{label}]")
            print(f"  {'-' * 100}")
            print(f"  {'Seed':<12} {'LR':<12} {'Mean':>8} {'Std':>8}  Values")
            print(f"  {'-' * 100}")

            for i, result in enumerate(per_seed_results):
                seed_name = seeds[i] if i < len(seeds) else str(i)

                if result is None:
                    print(f"  {seed_name:<12} {'N/A':<12}")
                else:
                    vals_str = "[" + ", ".join(f"{v:.4f}" for v in result["vals"]) + "]"
                    print(
                        f"  {seed_name:<12} "
                        f"{result['lr']:<12} "
                        f"{result['mean']:>8.4f} "
                        f"{result['std']:>8.4f}  "
                        f"{vals_str}"
                    )

            representative_accs = [x["mean"] for x in valid]
            cross_mean = float(np.mean(representative_accs))
            cross_std = float(np.std(representative_accs))
            accs_str = "[" + ", ".join(f"{v:.4f}" for v in representative_accs) + "]"

            print(f"  {'-' * 100}")
            print(f"  {'Cross-Seed':<24} {'Mean':>8} {'Std':>8}  Representative Accs")
            print(f"  {'':<24} {cross_mean:>8.4f} {cross_std:>8.4f}  {accs_str}")


# =============================================================================
# Plot
# =============================================================================

def plot(
    init_data_dict,
    init_config_dict,
    presented_keys,
    aug_key,
    save_path=None,
    only_lr=None,
    plot_rhos=None,
    plot_xticks=None,
    ylim=None,
):
    all_rhos = get_all_rhos(init_data_dict, aug_key)

    if plot_rhos is not None:
        plot_rhos = set(float(x) for x in plot_rhos)
        all_rhos = [rho for rho in all_rhos if rho in plot_rhos]

    fig, ax = plt.subplots(figsize=(8, 5))

    for init_key in presented_keys:
        if init_key not in init_data_dict:
            continue
        if init_key not in init_config_dict:
            continue

        label, color = init_config_dict[init_key]
        data_list = init_data_dict[init_key]

        xs = []
        means = []
        stds = []

        for rho in all_rhos:
            rho_str = str(float(rho))
            representative_accs = []

            for data in data_list:
                if data is None:
                    continue

                if aug_key not in data:
                    continue

                if rho_str not in data[aug_key]:
                    continue

                mean_val, _, _ = get_representative_acc(
                    data[aug_key][rho_str],
                    only_lr=only_lr,
                )

                if mean_val is None:
                    continue

                representative_accs.append(mean_val)

            if not representative_accs:
                continue

            xs.append(rho / 100.0)
            means.append(float(np.mean(representative_accs)))
            stds.append(float(np.std(representative_accs)))

        if not xs:
            continue

        xs = np.array(xs)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            xs,
            means * 100,
            marker="o",
            markersize=5,
            linewidth=2,
            label=label,
            color=color,
        )

        ax.fill_between(
            xs,
            (means - stds) * 100,
            (means + stds) * 100,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=15, labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=15, labelpad=8)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x * 100:.4g}")
    )

    if plot_xticks is not None:
        ax.set_xticks([float(x) / 100.0 for x in plot_xticks])
    else:
        ax.set_xticks([float(x) / 100.0 for x in all_rhos])

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(38.5, 98.0)

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot and print stats for random, ImageNet, and SimCLR initializations."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="../../exp_results/classification_hard",
        help="Base directory containing cold_start_random, cold_start_imagenet, cold_start_simclr.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["random10", "random24", "random38", "random42", "random57"],
        help="Seed prefixes used in JSON filenames.",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="bs16_ep20.json",
        help="Downstream suffix, e.g. bs16_ep20.json.",
    )

    parser.add_argument(
        "--aug_key",
        type=str,
        default="aug4",
        help="Augmentation key inside JSON, e.g. aug4, aug3, no_aug.",
    )

    parser.add_argument(
        "--portions",
        type=float,
        nargs="*",
        default=None,
        help="Only print stats for these rho values. Default: all.",
    )

    parser.add_argument(
        "--only_lr",
        type=str,
        default=None,
        help="Use fixed downstream LR instead of best LR per JSON.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default="./init_comparison.png",
        help="Figure output path. Use empty string '' to show interactively.",
    )

    parser.add_argument(
        "--plot_rhos",
        type=float,
        nargs="+",
        default=[2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Rho values included in plot.",
    )

    parser.add_argument(
        "--plot_xticks",
        type=float,
        nargs="+",
        default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="X-axis tick labels.",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits, e.g. --ylim 80 100.",
    )

    # Selected SimCLR setting for plotting only
    parser.add_argument(
        "--simclr_lr",
        type=str,
        default="0.0002",
        help="Selected SimCLR pretraining LR for plotting.",
    )

    parser.add_argument(
        "--simclr_bs",
        type=int,
        default=256,
        help="Selected SimCLR pretraining batch size for plotting.",
    )

    parser.add_argument(
        "--simclr_ep",
        type=int,
        default=500,
        help="Selected SimCLR pretraining epoch for plotting.",
    )

    args = parser.parse_args()

    if args.save == "":
        args.save = None

    # -------------------------------------------------------------------------
    # Scan all SimCLR JSON files
    # -------------------------------------------------------------------------
    simclr_records = scan_simclr_jsons(args.base_dir)
    unique_simclr_settings = get_unique_simclr_settings(simclr_records)

    # -------------------------------------------------------------------------
    # Config dictionary
    # -------------------------------------------------------------------------
    init_config_dict = dict(BASE_INIT_CONFIGS)

    for setting in unique_simclr_settings:
        simclr_lr = setting["simclr_lr"]
        simclr_bs = setting["simclr_bs"]
        simclr_ep = setting["simclr_ep"]

        simclr_key = make_simclr_init_key(simclr_lr, simclr_bs, simclr_ep)
        simclr_label = make_simclr_label(simclr_lr, simclr_bs, simclr_ep)

        init_config_dict[simclr_key] = (simclr_label, SIMCLR_COLOR)

    selected_simclr_key = make_simclr_init_key(
        args.simclr_lr,
        args.simclr_bs,
        args.simclr_ep,
    )

    selected_simclr_label = make_simclr_label(
        args.simclr_lr,
        args.simclr_bs,
        args.simclr_ep,
    )

    init_config_dict["simclr_selected_for_plot"] = (
        selected_simclr_label,
        SIMCLR_COLOR,
    )

    # -------------------------------------------------------------------------
    # Load Random and ImageNet
    # -------------------------------------------------------------------------
    terminal_data_dict = {}
    terminal_presented_keys = []

    plot_data_dict = {}
    plot_presented_keys = []

    for init_key in ["random", "imagenet"]:
        paths = get_standard_json_paths(
            base_dir=args.base_dir,
            init_key=init_key,
            seeds=args.seeds,
            suffix=args.suffix,
        )

        data_list = load_data_list(
            paths=paths,
            seeds=args.seeds,
            title=f"Loading init_type='{init_key}'",
        )

        if any(d is not None for d in data_list):
            terminal_data_dict[init_key] = data_list
            terminal_presented_keys.append(init_key)

            plot_data_dict[init_key] = data_list
            plot_presented_keys.append(init_key)

    # -------------------------------------------------------------------------
    # Load ALL parsed SimCLR settings for terminal stats
    # -------------------------------------------------------------------------
    for setting in unique_simclr_settings:
        simclr_lr = setting["simclr_lr"]
        simclr_bs = setting["simclr_bs"]
        simclr_ep = setting["simclr_ep"]

        simclr_key = make_simclr_init_key(simclr_lr, simclr_bs, simclr_ep)

        paths = get_simclr_json_paths(
            base_dir=args.base_dir,
            seeds=args.seeds,
            suffix=args.suffix,
            simclr_lr=simclr_lr,
            simclr_bs=simclr_bs,
            simclr_ep=simclr_ep,
            simclr_records=simclr_records,
        )

        data_list = load_data_list(
            paths=paths,
            seeds=args.seeds,
            title=f"Loading SimCLR setting for terminal stats: {init_config_dict[simclr_key][0]}",
        )

        if any(d is not None for d in data_list):
            terminal_data_dict[simclr_key] = data_list
            terminal_presented_keys.append(simclr_key)

    # -------------------------------------------------------------------------
    # Load SELECTED SimCLR setting for plotting only
    # -------------------------------------------------------------------------
    selected_paths = get_simclr_json_paths(
        base_dir=args.base_dir,
        seeds=args.seeds,
        suffix=args.suffix,
        simclr_lr=args.simclr_lr,
        simclr_bs=args.simclr_bs,
        simclr_ep=args.simclr_ep,
        simclr_records=simclr_records,
    )

    selected_data_list = load_data_list(
        paths=selected_paths,
        seeds=args.seeds,
        title=f"Loading selected SimCLR setting for plotting: {selected_simclr_label}",
    )

    if any(d is not None for d in selected_data_list):
        plot_data_dict["simclr_selected_for_plot"] = selected_data_list
        plot_presented_keys.append("simclr_selected_for_plot")

    # -------------------------------------------------------------------------
    # Print final terminal stats
    # This is the block you want:
    # Random, ImageNet, and ALL SimCLR settings under the same rho section.
    # -------------------------------------------------------------------------
    print("")
    print("=" * 120)
    print("FINAL TERMINAL STATS: Random, ImageNet, and ALL parsed SimCLR settings")
    print("=" * 120)

    print_stats(
        init_data_dict=terminal_data_dict,
        init_config_dict=init_config_dict,
        presented_keys=terminal_presented_keys,
        seeds=args.seeds,
        aug_key=args.aug_key,
        filter_rhos=args.portions,
        only_lr=args.only_lr,
    )

    # -------------------------------------------------------------------------
    # Plot only Random, ImageNet, and selected SimCLR
    # -------------------------------------------------------------------------
    print("")
    print("=" * 120)
    print("PLOTTING SELECTED CONFIG")
    print("=" * 120)
    print(f"Selected SimCLR: {selected_simclr_label}")
    print("=" * 120)

    plot(
        init_data_dict=plot_data_dict,
        init_config_dict=init_config_dict,
        presented_keys=plot_presented_keys,
        aug_key=args.aug_key,
        save_path=args.save,
        only_lr=args.only_lr,
        plot_rhos=args.plot_rhos,
        plot_xticks=args.plot_xticks,
        ylim=args.ylim,
    )


if __name__ == "__main__":
    main()