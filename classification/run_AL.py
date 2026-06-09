import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR
from classification.utils.train_eval import train_model

from classification.AL_strategy.uncertainty import conf, entropy, margin
from classification.AL_strategy.diversity_correct import coreset, typiclust   # 正確版 + TypiClust
from classification.AL_strategy.hybrid_correct import badge, cluster_margin    # 正確版 + Cluster-Margin


"""
=================================================================================
ACTIVE LEARNING — 4.4 protocol (lr sweep + best-val model selects)
=================================================================================
每個 --seed = 一條獨立 AL 軌跡，初始 labeled pool 由該 seed 隨機選取（5 個 seed →
5 條軌跡 → mean±std 取自 5 seeds）。

--lr_schedule（每個 portion 的 lr 做法）：
  - 'sweep'（預設，option A）：每個 portion 對候選 lr（--lr_grid 或 lr_grid_for(portion)）
    各自 fresh-init 訓練，**用 validation loss 最低的 model 當「選取器」去選下一批**，
    同時把每個 lr 的 test_acc 都存進 JSON（aggregate 取 best-lr 仍一致）。挑 lr 用 val
    而非 test，避免 test leakage。
  - 'coldstart'：每個 portion 用 θ² cold-start 查到的單一 best lr（不重掃，便宜）。
  - 'fixed'：整條用單一 --lr。

每個 portion 的 labeled set 另存到 labeled_ids/ 供 reproduce 與 Chapter-5 視覺化。
=================================================================================
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--AL_strategy', type=str,
                        choices=['random', 'conf', 'entropy', 'margin', 'coreset', 'badge',
                                 'typiclust', 'cluster_margin'],
                        required=True)

    # AL related setup
    parser.add_argument('--portion_start', type=float, required=True)
    parser.add_argument('--portion_end', type=float, required=True)
    parser.add_argument('--portion_interval', type=float, required=True)   
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)

    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, 
                        choices=['random', 'imagenet', 'simclr', 'auto_encoder'], 
                        required=True)   
    parser.add_argument('--simclr_path', type=str, default=None)
    
    # Training hyperparameters (aligned with run_first_iter.py)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="Used when --lr_schedule fixed, or as fallback if coldstart ref is missing.")
    parser.add_argument('--lr_schedule', type=str, choices=['sweep', 'coldstart', 'fixed'], default='sweep',
                        help="'sweep'(option A): 每 portion 掃多個 lr、用 val 最佳的 model 選下一批；"
                             "'coldstart': 每 portion 用 cold-start 查到的單一 best lr；'fixed': 單一 --lr。")
    parser.add_argument('--lr_grid', type=str, default=None,
                        help="sweep 模式的下游 lr 候選（空白分隔）。未指定則用 lr_grid_for(portion) 的 per-portion 預設。")
    parser.add_argument('--coldstart_ref', type=str,
                        default='./classification/exp_results/classification_hard/cold_start_simclr/random42_bs16.json',
                        help="coldstart 模式用：theta^2 cold-start result json（path relative to repo root）。")
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false', default=True)
    parser.add_argument('--aug_factor', type=int, default=4)   
    parser.add_argument('--flip_type', type=str, default='horizontal')
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained):
    model = get_resnet18_classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def initialize_simclr_model(num_classes, simclr_path):
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
    return model


def load_best_lr_schedule(ref_path, aug_key):
    """Read a theta^2 cold-start result json and return {portion(float): best_lr(str)},
    where best_lr maximises mean accuracy at that portion. Empty dict if unavailable."""
    if not os.path.isfile(ref_path):
        print(f"Warning: coldstart_ref not found: {ref_path} -> fall back to fixed --lr")
        return {}
    with open(ref_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if aug_key not in d:
        print(f"Warning: aug_key '{aug_key}' not in {ref_path} -> fall back to fixed --lr")
        return {}
    sched = {}
    for p, lr_dict in d[aug_key].items():
        best_lr, best_mean = None, -1.0
        for lr, vals in lr_dict.items():
            acc = vals["acc"] if isinstance(vals, dict) else vals
            m = float(np.mean(acc)) if len(acc) else -1.0
            if m > best_mean:
                best_mean, best_lr = m, lr
        if best_lr is not None:
            sched[float(p)] = best_lr
    return sched


def lr_for_portion(portion, sched, fallback):
    """Best lr of the nearest available portion in the schedule; fallback if empty."""
    if not sched:
        return float(fallback)
    nearest = min(sched.keys(), key=lambda q: abs(q - portion))
    return float(sched[nearest])


def lr_grid_for(portion):
    """sweep 模式：每個 portion 的下游 lr 候選網格（統一）。
    依 θ² cold-start 參考：best-lr 幾乎都是 5e-5 或 1e-4，5e-4/1e-5 從來不是最佳；
    AL-entropy 低 portion 另需 3e-4。故取 3e-5 5e-5 1e-4 3e-4，讓最佳值落在內部、不卡邊緣。"""
    return ["3e-5", "5e-5", "1e-4", "3e-4"]


def coldstart_best_lr(seed, portion, exp_path, task_type, aug_key,
                      cfg="simclr_lr0.0002_simclr_bs256_simclr_ep500"):
    """讀 θ² cold-start 的 per-seed 檔，回傳該 (seed, portion) 的 best downstream lr（字串）。
    用於 AL 初始步：與 Random baseline 一致（同 seed → 同 2.5% 子集）。找不到回 None → 退回 sweep。"""
    f = os.path.join(exp_path, f"classification_{task_type}", "cold_start_simclr",
                     f"random{seed}_bs16_ep20_{cfg}.json")
    if not os.path.isfile(f):
        return None
    try:
        d = json.load(open(f))
    except Exception:
        return None
    pk = str(float(portion))
    if aug_key not in d or pk not in d[aug_key]:
        return None
    best_lr, best = None, -1.0
    for lr, vals in d[aug_key][pk].items():
        acc = vals["acc"] if isinstance(vals, dict) else vals
        if len(acc) and float(np.mean(acc)) > best:
            best, best_lr = float(np.mean(acc)), lr
    return best_lr


def build_model(pretrained_weights, num_classes, simclr_path=None):
    """依 init 種類建立全新的分類模型（sweep 時每個 lr 都要 fresh init）。"""
    if pretrained_weights == 'random':
        return initialize_model(num_classes, False)
    elif pretrained_weights == 'simclr':
        return initialize_simclr_model(num_classes, simclr_path)
    elif pretrained_weights == 'imagenet':
        return initialize_model(num_classes, True)
    elif pretrained_weights == 'auto_encoder':
        raise NotImplementedError("Auto encoder not implemented yet")
    raise NotImplementedError(f"Pretrained weights {pretrained_weights} not implemented")


def save_compact_json(data, file_path):
    """Save JSON with compact list formatting"""
    def format_dict(d, indent=0):
        lines = []
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            comma = '' if is_last else ','
            
            if isinstance(value, dict):
                lines.append('  ' * indent + f'"{key}": {{')
                lines.append(format_dict(value, indent + 1))
                lines.append('  ' * indent + '}' + comma)
            elif isinstance(value, list):
                # Keep list on single line
                list_str = '[' + ', '.join(str(x) for x in value) + ']'
                lines.append('  ' * indent + f'"{key}": {list_str}' + comma)
            else:
                lines.append('  ' * indent + f'"{key}": {json.dumps(value)}' + comma)
        
        return '\n'.join(lines)
    
    json_str = '{\n' + format_dict(data, 1) + '\n}'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def main():
    args = parse_arguments()
    random.seed(args.seed) # this will be realted to only how the initial labeled pool be selected
    
    # Determine task configuration
    task_config = {
        'easy': (2, './ds/classification/two_class'),
        'medium': (4, './ds/classification/four_class'),
        'hard': (7, './ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    # Fix batch size to 16 (aligned with run_first_iter.py)
    batch_size = 16
    
    # Generate file name (aligned with run_first_iter.py format)
    file_name = f"{args.AL_strategy}_seed{args.seed}_bs{batch_size}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
    
    print(f'Exp name: {file_name}')
    print(f'AL Strategy: {args.AL_strategy}')
    print(f'Portion range: {args.portion_start}% to {args.portion_end}% (interval: {args.portion_interval}%)')
    print(f'Learning Rate: {args.lr} (FIXED across all portions)')
    print(f'Batch Size: {batch_size}')
    print('-' * 60)
    
    # Generate aug key (aligned with run_first_iter.py)
    if not args.data_aug:
        aug_key = "no_aug"
    elif args.aug_factor == 2:
        aug_key = f"aug{args.aug_factor}_{args.flip_type}"
    else:
        aug_key = f"aug{args.aug_factor}"
    
    print(f'Aug config key: {aug_key}')
    
    # Initialize label/unlabel indices
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    label_idx = []
    unlabeled_idx = list(range(tot_num_train))
    
    # Prepare save path
    save_path = os.path.join(args.exp_path, 
                             f"classification_{args.task_type}", 
                             f"AL_{args.pretrained_weights}")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    # Load or create results file
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error in {file_path}: {e}")
            print("Creating new data structure...")
            data = {}
    else:
        data = {}
    
    # Initialize data structure
    if aug_key not in data:
        data[aug_key] = {}

    # Per-portion lr schedule (4.4 protocol)
    if args.lr_schedule == 'coldstart':
        lr_sched = load_best_lr_schedule(args.coldstart_ref, aug_key)
        print(f"LR schedule: coldstart (nearest-portion best lr), {len(lr_sched)} ref portions")
    else:
        lr_sched = {}
        print(f"LR schedule: fixed lr={args.lr}")

    # Dedicated labeled-id dump (for reproduction + Chapter-5 visualisation)
    labeled_dir = os.path.join(save_path, "labeled_ids")
    os.makedirs(labeled_dir, exist_ok=True)
    labeled_file = os.path.join(labeled_dir, f"{args.AL_strategy}_seed{args.seed}_bs{batch_size}.json")
    labeled_log = {}
    if os.path.isfile(labeled_file):
        try:
            with open(labeled_file, "r", encoding="utf-8") as f:
                labeled_log = json.load(f)
        except json.JSONDecodeError:
            labeled_log = {}

    # Active Learning Loop
    last_trained_model = None
    
    for portion in np.arange(args.portion_start, args.portion_end, args.portion_interval):
        print('\n' + '=' * 60)
        print(f'PORTION: {portion}%')
        print('=' * 60)
        
        portion_key = str(float(portion))

        # ===== 決定本 portion 的候選下游 lr =====
        if args.lr_schedule == 'sweep':
            if portion == args.portion_start:
                # 初始步：與 Random baseline 一致 → 用該 seed 在此 portion 的 cold-start best lr（不重掃）。
                # 後續 portion 因 AL 選樣改變 labeled set，optimal lr 會不同 → 自己 sweep。
                blr = coldstart_best_lr(args.seed, portion, args.exp_path, args.task_type, aug_key)
                if blr:
                    cand_lrs = [blr]
                    print(f'初始 ρ={portion}%：用 θ² cold-start seed{args.seed} best lr={blr}（與 Random 一致，不掃）')
                else:
                    cand_lrs = args.lr_grid.split() if args.lr_grid else lr_grid_for(portion)
                    print(f'初始 ρ={portion}%：cold-start 無 seed{args.seed} 資料 → 退回 sweep {cand_lrs}')
            else:
                cand_lrs = args.lr_grid.split() if args.lr_grid else lr_grid_for(portion)
        elif args.lr_schedule == 'coldstart':
            cand_lrs = [str(lr_for_portion(portion, lr_sched, args.lr))]
        else:  # fixed
            cand_lrs = [str(args.lr)]
        print(f'Candidate lrs for ρ={portion}%: {cand_lrs}')

        if portion_key not in data[aug_key]:
            data[aug_key][portion_key] = {}
        
        # ===== Select Data to Label =====
        target_num = round(tot_num_train * portion / 100)
        num_to_label = target_num - len(label_idx)
        
        if portion == args.portion_start:
            # First iteration: random sampling
            print(f'===== First Iteration: Random Sample {num_to_label} samples =====')
            to_label_idx = random.sample(unlabeled_idx, num_to_label)
        else:
            # Subsequent iterations: use AL strategy
            print(f'===== AL Strategy: {args.AL_strategy} - Select {num_to_label} samples =====')
            
            if args.AL_strategy == 'random':   # passive baseline：每步隨機選，不看 model
                to_label_idx = random.sample(unlabeled_idx, num_to_label)
            elif args.AL_strategy == 'conf':
                to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'margin':
                to_label_idx, _ = margin(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'entropy':
                to_label_idx = entropy(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)    
            elif args.AL_strategy == 'coreset':
                # 傳入目前已標註集 label_idx 當 k-center 的初始中心（正確 coreset）
                to_label_idx = coreset(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device, label_idx)
            elif args.AL_strategy == 'typiclust':
                # 低預算 diversity：含已標註集 → 群覆蓋偏好未標註區（density-based）
                to_label_idx = typiclust(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device, label_idx)
            elif args.AL_strategy == 'badge':
                to_label_idx = badge(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'cluster_margin':
                to_label_idx = cluster_margin(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            else:
                raise NotImplementedError(f"AL strategy {args.AL_strategy} not implemented")
        
        # Update label and unlabeled indices
        label_idx.extend(to_label_idx)
        unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
        
        print(f"Selected {len(to_label_idx)} samples")
        print(f"Total labeled: {len(label_idx)} | Remaining unlabeled: {len(unlabeled_idx)}")
        
        # Sanity check
        if len(label_idx) != len(set(label_idx)):
            raise ValueError("Duplicate indices in label_idx!")
        if len(unlabeled_idx) != len(set(unlabeled_idx)):
            raise ValueError("Duplicate indices in unlabeled_idx!")

        # ===== Dump per-portion labeled ids (reproduction + Chapter-5 viz) =====
        # selected = newly chosen this step; cumulative = full labeled set at this portion.
        labeled_log[portion_key] = {
            "lrs_swept": cand_lrs,
            "n_cumulative": len(label_idx),
            "selected": list(to_label_idx),
            "cumulative": label_idx.copy(),
        }
        save_compact_json(
            {k: labeled_log[k] for k in sorted(labeled_log, key=float)}, labeled_file)
        print(f"Labeled ids saved -> {labeled_file}")

        # ===== Load Data once (所有候選 lr 共用同一個 labeled set) =====
        if args.data_aug:
            data_loaders, dataset_sizes = get_data(
                data_dir, label_idx, batch_size,
                data_aug=True, aug_factor=args.aug_factor, flip_type=args.flip_type
            )
        else:
            data_loaders, dataset_sizes = get_data(data_dir, label_idx, batch_size, data_aug=False)
        print(f"Dataset sizes: {dataset_sizes}")

        # ===== Sweep 候選 lr：各自 fresh-init 訓練，用 val loss 最低者當「選取器」(option A) =====
        criterion = nn.CrossEntropyLoss()
        best = {"val": float('inf'), "model": None, "lr": None, "acc": None}
        for lr_str in cand_lrs:
            lr_f = float(lr_str)
            lr_key = str(lr_f)
            print('=' * 50)
            print(f'===== ρ={portion}% | init={args.pretrained_weights} | seed={args.seed} | '
                  f'aug={aug_key} | lr={lr_f} | bs={batch_size} | ep={args.epoch} | n={len(label_idx)} =====')
            model = build_model(args.pretrained_weights, num_classes, args.simclr_path)
            if args.weight_decay is not None:
                optimizer_ = optim.AdamW(model.parameters(), lr=lr_f, weight_decay=args.weight_decay)
            else:
                optimizer_ = optim.AdamW(model.parameters(), lr=lr_f)
            lr_scheduler_ = lr_scheduler.LinearLR(
                optimizer_, start_factor=1.0, end_factor=0.0, total_iters=args.epoch)

            trained_model, test_acc, val_loss = train_model(
                model, args.device, data_loaders, dataset_sizes,
                criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch
            )
            test_acc = round(test_acc, 4)
            print(f'  -> lr={lr_f}: val_loss={val_loss:.4f}  test_acc={test_acc}')

            data[aug_key][portion_key].setdefault(lr_key, {"acc": [], "labeled_idx": []})
            data[aug_key][portion_key][lr_key]["acc"].append(test_acc)
            data[aug_key][portion_key][lr_key]["labeled_idx"].append(label_idx.copy())

            if val_loss < best["val"]:
                best = {"val": val_loss, "model": trained_model, "lr": lr_f, "acc": test_acc}

        # val 最佳的 model 當下一步的選取器（option A 核心：用最好的 model 去選 label）
        last_trained_model = best["model"]
        print(f"Best-val lr @ ρ={portion}%: {best['lr']} "
              f"(val_loss={best['val']:.4f}, test_acc={best['acc']}) → 用它 select 下一批")

        # ===== Save Results =====
        # Sort portions AND lrs within each aug configuration
        sorted_data = {}
        for aug_k in sorted(data.keys()):
            sorted_portions = {}
            for portion_k in sorted(data[aug_k].keys(), key=float):
                sorted_lrs = {}
                for lr_k in sorted(data[aug_k][portion_k].keys(), key=float):
                    sorted_lrs[lr_k] = data[aug_k][portion_k][lr_k]
                sorted_portions[portion_k] = sorted_lrs
            sorted_data[aug_k] = sorted_portions
        
        # Save with compact list format
        save_compact_json(sorted_data, file_path)
        print(f'Result saved to {file_path}!')
    
    print('\n' + '=' * 60)
    print('Active Learning Completed!')
    print('=' * 60)


if __name__ == "__main__":
    main()