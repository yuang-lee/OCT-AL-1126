import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Regex
# =============================================================================

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
# Hardcoded SimCLR pretraining LR by SimCLR batch size
# =============================================================================

SIMCLR_BS_TO_LR = {
    16: "0.0001",
    32: "0.00015",   # sqrt scaling value was approximately 0.0001414
    64: "0.0002",
    128: "0.0003",   # sqrt scaling value was approximately 0.0002828
    256: "0.0004",
}


# =============================================================================
# Parsing helpers
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
    """
    Parse one SimCLR JSON filename.
    """
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


def format_number_for_filename(x):
    """
    Make values like 10.0 become 10, and 2.5 remain 2p5 for safe filenames.
    """
    x = float(x)

    if x.is_integer():
        s = str(int(x))
    else:
        s = str(x)

    return s.replace(".", "p")


# =============================================================================
# JSON scanning
# =============================================================================

def scan_simclr_jsons(base_dir):
    """
    Scan cold_start_simclr/*.json and parse all SimCLR configs.
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
        raise FileNotFoundError(f"Directory not found: {simclr_dir}")

    json_files = sorted(
        f for f in os.listdir(simclr_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(simclr_dir, f))
    )

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
        )
    )

    print("")
    print("=" * 140)
    print("Parsed SimCLR JSON files")
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

    unique_bs = sorted({int(float(r["simclr_bs"])) for r in records})
    unique_ep = sorted({int(float(r["simclr_ep"])) for r in records})
    unique_lr = sorted({str(r["simclr_lr"]) for r in records}, key=float)

    print("")
    print("=" * 140)
    print("Parsed SimCLR grid summary")
    print("=" * 140)
    print(f"SimCLR_LR values : {unique_lr}")
    print(f"SimCLR_BS values : {unique_bs}")
    print(f"SimCLR_EP values : {unique_ep}")

    if unparsed:
        print("")
        print("=" * 140)
        print("Unparsed JSON files under cold_start_simclr")
        print("=" * 140)

        for filename in unparsed:
            print(f"  {filename}")

    return records


def filter_records(
    records,
    seeds,
    suffix,
    simclr_lr_by_bs,
):
    """
    Keep only records matching:
    - selected seed list
    - selected downstream suffix, e.g. bs16_ep20.json
    - hardcoded SimCLR LR corresponding to each SimCLR_BS

    Example:
        simclr_bs=16  -> simclr_lr=0.0001
        simclr_bs=32  -> simclr_lr=0.00015
        simclr_bs=64  -> simclr_lr=0.0002
        simclr_bs=128 -> simclr_lr=0.0003
        simclr_bs=256 -> simclr_lr=0.0004
    """
    suffix_info = parse_standard_suffix(suffix)
    downstream_bs = suffix_info["downstream_bs"]
    downstream_ep = suffix_info["downstream_ep"]

    seed_set = set(seeds)

    filtered = []

    for r in records:
        if r["seed"] not in seed_set:
            continue

        if str(r["downstream_bs"]) != str(downstream_bs):
            continue

        if str(r["downstream_ep"]) != str(downstream_ep):
            continue

        simclr_bs = int(float(r["simclr_bs"]))

        if simclr_bs not in simclr_lr_by_bs:
            continue

        expected_simclr_lr = simclr_lr_by_bs[simclr_bs]

        if not safe_float_equal(r["simclr_lr"], expected_simclr_lr):
            continue

        filtered.append(r)

    return filtered


# =============================================================================
# Accuracy selection
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def extract_accuracy_from_record(record, aug_key, portion, only_lr):
    """
    For one JSON record:
    - read data[aug_key][portion]
    - choose best downstream LR or fixed downstream LR
    - return mean accuracy for that selected LR
    """
    data = load_json(record["path"])

    portion_key = str(float(portion))

    if aug_key not in data:
        return None, None, []

    if portion_key not in data[aug_key]:
        return None, None, []

    rho_data = data[aug_key][portion_key]

    mean_acc, selected_lr, vals = get_representative_acc(
        rho_data=rho_data,
        only_lr=only_lr,
    )

    return mean_acc, selected_lr, vals


# =============================================================================
# Heatmap data construction
# =============================================================================

def build_heatmap_matrix(
    records,
    seeds,
    suffix,
    simclr_lr_by_bs,
    aug_key,
    portion,
    only_lr,
    batch_sizes=None,
    epochs=None,
):
    """
    Build heatmap matrix.

    Important logic:
    For each SimCLR_BS x SimCLR_EP cell:

        1. Select JSON files whose SimCLR LR matches the hardcoded LR
           corresponding to that SimCLR_BS.
        2. For each seed, read that seed's JSON.
        3. Within that seed and portion, choose:
              - the best downstream LR by mean accuracy, if only_lr is None; or
              - the fixed downstream LR, if only_lr is provided.
        4. Take the mean accuracy under that selected LR for that seed.
        5. Average these per-seed selected-LR means across seeds.

    Therefore, each cell is:

        mean_over_seeds(
            best_lr_mean_accuracy_within_each_seed
        )

    It does NOT pool all downstream LR values across seeds.
    """
    filtered = filter_records(
        records=records,
        seeds=seeds,
        suffix=suffix,
        simclr_lr_by_bs=simclr_lr_by_bs,
    )

    if not filtered:
        raise RuntimeError(
            "No SimCLR JSON records matched the requested "
            "seeds/suffix/hardcoded simclr_lr_by_bs."
        )

    if batch_sizes is None:
        batch_sizes = sorted({int(float(r["simclr_bs"])) for r in filtered})
    else:
        batch_sizes = [int(x) for x in batch_sizes]

    if epochs is None:
        epochs = sorted({int(float(r["simclr_ep"])) for r in filtered})
    else:
        epochs = [int(x) for x in epochs]

    # -------------------------------------------------------------------------
    # Organize records by (SimCLR_BS, SimCLR_EP)
    # -------------------------------------------------------------------------
    records_by_cell = {
        (bs, ep): []
        for bs in batch_sizes
        for ep in epochs
    }

    for r in filtered:
        bs = int(float(r["simclr_bs"]))
        ep = int(float(r["simclr_ep"]))

        if bs not in batch_sizes:
            continue
        if ep not in epochs:
            continue

        records_by_cell[(bs, ep)].append(r)

    matrix = np.full(
        shape=(len(batch_sizes), len(epochs)),
        fill_value=np.nan,
        dtype=float,
    )

    cell_details = {
        (bs, ep): []
        for bs in batch_sizes
        for ep in epochs
    }

    # -------------------------------------------------------------------------
    # Compute each cell:
    # per seed -> selected downstream LR mean -> mean across seeds
    # -------------------------------------------------------------------------
    for i, bs in enumerate(batch_sizes):
        for j, ep in enumerate(epochs):
            per_seed_means = []

            seed_to_record = {}

            for r in records_by_cell[(bs, ep)]:
                seed_to_record[r["seed"]] = r

            for seed in seeds:
                if seed not in seed_to_record:
                    cell_details[(bs, ep)].append(
                        {
                            "seed": seed,
                            "available": False,
                            "selected_downstream_lr": None,
                            "values": [],
                            "mean_acc": None,
                            "path": None,
                        }
                    )
                    continue

                r = seed_to_record[seed]

                mean_acc, selected_lr, vals = extract_accuracy_from_record(
                    record=r,
                    aug_key=aug_key,
                    portion=portion,
                    only_lr=only_lr,
                )

                if mean_acc is None:
                    cell_details[(bs, ep)].append(
                        {
                            "seed": seed,
                            "available": False,
                            "selected_downstream_lr": None,
                            "values": [],
                            "mean_acc": None,
                            "path": r["path"],
                        }
                    )
                    continue

                per_seed_means.append(mean_acc)

                cell_details[(bs, ep)].append(
                    {
                        "seed": seed,
                        "available": True,
                        "selected_downstream_lr": selected_lr,
                        "values": vals,
                        "mean_acc": mean_acc,
                        "path": r["path"],
                    }
                )

            if per_seed_means:
                matrix[i, j] = float(np.mean(per_seed_means)) * 100.0

    # -------------------------------------------------------------------------
    # Print cell details
    # -------------------------------------------------------------------------
    print("")
    print("=" * 150)
    print("Heatmap cell details")
    print("=" * 150)
    print(f"Aug key       : {aug_key}")
    print(f"Portion       : {portion}%")
    print(f"SimCLR LR map : {simclr_lr_by_bs}")
    print(
        "Downstream LR : "
        + (f"fixed {only_lr}" if only_lr is not None else "best downstream LR within each seed")
    )
    print("")
    print("Each cell = mean over seeds of the selected-LR mean accuracy within each seed.")
    print("-" * 150)
    print(
        f"{'SimCLR_BS':>10} "
        f"{'SimCLR_EP':>10} "
        f"{'SimCLR_LR':>12} "
        f"{'N_seed':>8} "
        f"{'Cell Acc(%)':>12} "
        f"Per-seed selected downstream LR and mean"
    )
    print("-" * 150)

    for i, bs in enumerate(batch_sizes):
        for j, ep in enumerate(epochs):
            details = cell_details[(bs, ep)]
            valid_details = [d for d in details if d["available"]]
            simclr_lr_for_bs = simclr_lr_by_bs.get(bs, "NA")

            if len(valid_details) == 0:
                print(
                    f"{bs:>10} "
                    f"{ep:>10} "
                    f"{simclr_lr_for_bs:>12} "
                    f"{0:>8} "
                    f"{'NA':>12} "
                    f"NA"
                )
                continue

            per_seed_text = "; ".join(
                f"{d['seed']}: lr={d['selected_downstream_lr']}, "
                f"mean={d['mean_acc']:.4f}, "
                f"vals=[{', '.join(f'{v:.4f}' for v in d['values'])}]"
                for d in valid_details
            )

            print(
                f"{bs:>10} "
                f"{ep:>10} "
                f"{simclr_lr_for_bs:>12} "
                f"{len(valid_details):>8} "
                f"{matrix[i, j]:>12.2f} "
                f"{per_seed_text}"
            )

    print("-" * 150)

    return matrix, batch_sizes, epochs


# =============================================================================
# Plot heatmap
# =============================================================================

def plot_heatmap(
    matrix,
    batch_sizes,
    epochs,
    save_path,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotate=True,
):
    """
    Plot SimCLR grid heatmap.

    x-axis = SimCLR epoch
    y-axis = SimCLR batch size
    color = accuracy (%)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("SimCLR Pretraining Epoch", fontsize=15, labelpad=8)
    ax.set_ylabel("SimCLR Pretraining Batch Size", fontsize=15, labelpad=8)

    ax.set_xticks(np.arange(len(epochs)))
    ax.set_yticks(np.arange(len(batch_sizes)))

    ax.set_xticklabels([str(x) for x in epochs])
    ax.set_yticklabels([str(x) for x in batch_sizes])

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    # Minor grid lines to separate cells
    ax.set_xticks(np.arange(-0.5, len(epochs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(batch_sizes), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)", fontsize=15, labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    if annotate:
        finite_vals = matrix[np.isfinite(matrix)]

        if finite_vals.size > 0:
            threshold = float(np.nanmean(finite_vals))
        else:
            threshold = 0.0

        for i in range(len(batch_sizes)):
            for j in range(len(epochs)):
                value = matrix[i, j]

                if np.isnan(value):
                    text = "NA"
                    text_color = "black"
                else:
                    text = f"{value:.2f}"
                    text_color = "white" if value < threshold else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=text_color,
                )

    # No figure title by request
    fig.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"\nHeatmap saved to: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot SimCLR batch-size × epoch heatmap for one data portion. "
            "SimCLR pretraining LR is hardcoded by SimCLR batch size."
        )
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="../../exp_results/classification_hard",
        help="Base directory containing cold_start_simclr.",
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
        "--portion",
        type=float,
        required=True,
        help="Data portion to plot. Example: --portion 10 or --portion 100.",
    )

    parser.add_argument(
        "--only_lr",
        type=str,
        default=None,
        help="Fixed downstream classifier LR. If omitted, use best downstream LR per JSON.",
    )

    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional batch-size order/filter for y-axis. Example: --batch_sizes 16 32 64 128 256.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional epoch order/filter for x-axis. Example: --epochs 10 20 50 100 200 500.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory to save heatmap figure.",
    )

    parser.add_argument(
        "--save_prefix",
        type=str,
        default="simclr_heatmap",
        help="Filename prefix. Portion is automatically appended.",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap. Default: viridis.",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional colorbar minimum accuracy percentage.",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional colorbar maximum accuracy percentage.",
    )

    parser.add_argument(
        "--no_annotate",
        action="store_true",
        default=False,
        help="Disable cell annotations.",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    portion_tag = format_number_for_filename(args.portion)
    aug_tag = args.aug_key

    save_path = os.path.join(
        args.save_dir,
        f"{args.save_prefix}_{aug_tag}_portion{portion_tag}_lr.png",
    )

    print("")
    print("=" * 140)
    print("Hardcoded SimCLR LR by batch size")
    print("=" * 140)
    for bs in sorted(SIMCLR_BS_TO_LR.keys()):
        print(f"SimCLR_BS={bs:>4} -> SimCLR_LR={SIMCLR_BS_TO_LR[bs]}")
    print("=" * 140)

    records = scan_simclr_jsons(args.base_dir)

    matrix, batch_sizes, epochs = build_heatmap_matrix(
        records=records,
        seeds=args.seeds,
        suffix=args.suffix,
        simclr_lr_by_bs=SIMCLR_BS_TO_LR,
        aug_key=args.aug_key,
        portion=args.portion,
        only_lr=args.only_lr,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
    )

    plot_heatmap(
        matrix=matrix,
        batch_sizes=batch_sizes,
        epochs=epochs,
        save_path=save_path,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annotate=not args.no_annotate,
    )


if __name__ == "__main__":
    main()