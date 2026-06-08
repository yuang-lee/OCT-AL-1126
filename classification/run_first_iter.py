import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.data import get_data, get_num_train
from model.resnet import get_resnet18_classifier
from model.simclr.resnet_simclr import ResNetSimCLR 
from utils.train_eval import train_model


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--portion', type=float, required=True) 
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)   
    parser.add_argument('--lr', type=float, default=5e-5) 
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--pretrained_weights', type=str, choices=['random', 'imagenet', 'simclr', 'auto_encoder'], default=None, required=True)   
    parser.add_argument('--simclr_path', type=str, default=None)
    parser.add_argument('--ask_saving', action='store_true', default=False)
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false', default=True)
    parser.add_argument('--aug_factor', type=int, default=4)   
    parser.add_argument('--flip_type', type=str, default='horizontal')
    # ColorJitter arguments
    parser.add_argument('--color_jitter', action='store_true', default=False,
                        help='Enable online ColorJitter augmentation on training set')
    parser.add_argument('--jitter_brightness', type=float, default=0.3,
                        help='ColorJitter brightness range (default: 0.3)')
    parser.add_argument('--jitter_contrast', type=float, default=0.3,
                        help='ColorJitter contrast range (default: 0.3)')

    return parser.parse_args()


def initialize_model(num_classes, pretrained):
    model = get_resnet18_classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def initialize_simclr_model(num_classes, simclr_path):
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    # print(model)
    
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
    return model


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
                list_str = '[' + ', '.join(str(x) for x in value) + ']'
                lines.append('  ' * indent + f'"{key}": {list_str}' + comma)
            else:
                lines.append('  ' * indent + f'"{key}": {json.dumps(value)}' + comma)
        
        return '\n'.join(lines)
    
    json_str = '{\n' + format_dict(data, 1) + '\n}'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def check_existing_results(file_path, aug_key, portion_key, lr_key, max_runs=5):
    """
    Check if the experiment has already been run enough times.
    Raises an error if the result list already has max_runs or more entries.
    """
    if not os.path.isfile(file_path):
        print(f"No existing results file found at {file_path}")
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return
    
    if aug_key in data:
        if portion_key in data[aug_key]:
            if lr_key in data[aug_key][portion_key]:
                existing_results = data[aug_key][portion_key][lr_key]
                num_existing = len(existing_results)
                
                if num_existing >= max_runs:
                    raise RuntimeError(
                        f"\n{'='*60}\n"
                        f"Experiment already completed!\n"
                        f"Configuration: aug={aug_key}, portion={portion_key}, lr={lr_key}\n"
                        f"Existing results: {existing_results}\n"
                        f"Number of runs: {num_existing}/{max_runs}\n"
                        f"File: {file_path}\n"
                        f"{'='*60}\n"
                    )
                else:
                    print(f"Found {num_existing}/{max_runs} existing results for this configuration")
            else:
                print(f"No results found for lr={lr_key}")
        else:
            print(f"No results found for portion={portion_key}")
    else:
        print(f"No results found for aug_key={aug_key}")


def main():
    args = parse_arguments()
    
    # Determine task configuration
    task_config = {
        'easy': (2, '../ds/classification/two_class'),
        'medium': (4, '../ds/classification/four_class'),
        'hard': (7, '../ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    # Fix batch size to 16
    args.batch_size = 16
    
    # Generate file name
    file_name = f"random{args.seed}_bs{args.batch_size}_ep{args.epoch}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
        
    print(f'Exp name: {file_name}')

    # ===== CHECK IF EXPERIMENT ALREADY COMPLETED =====
    if not args.data_aug:
        aug_key = "no_aug"
    elif args.aug_factor == 2:
        aug_key = f"aug{args.aug_factor}_{args.flip_type}"
    else:
        aug_key = f"aug{args.aug_factor}"

    # Append jitter suffix to aug_key if enabled
    if args.color_jitter:
        aug_key += f"_jitter_b{args.jitter_brightness}_c{args.jitter_contrast}"
    
    portion_key = str(float(args.portion))
    lr_key = str(args.lr)
    
    save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", f"cold_start_{args.pretrained_weights}")
    file_path = os.path.join(save_path, file_name)
    print(f"\nChecking existing results...")
    check_existing_results(file_path, aug_key, portion_key, lr_key, max_runs=3)
    print(f"Check passed. Proceeding with training...\n")
    # =================================================
    
    # Get training data indices
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    
    print(f'===== Load {args.portion}% Data =====')
    target_num = round(tot_num_train * args.portion / 100)
    
    random.seed(args.seed)
    unlabeled_idx = list(range(tot_num_train))
    label_idx = random.sample(unlabeled_idx, target_num)
    
    print(f"Number of labeled samples: {len(label_idx)}")
    
    # Load data
    if args.data_aug:
        if args.aug_factor is None:
            raise ValueError("aug_factor is required when data_aug is True")
        data_loaders, dataset_sizes = get_data(
            data_dir, label_idx, args.batch_size,
            data_aug=True,
            aug_factor=args.aug_factor,
            flip_type=args.flip_type,
            color_jitter=args.color_jitter,
            jitter_brightness=args.jitter_brightness,
            jitter_contrast=args.jitter_contrast,
        )
    else:
        data_loaders, dataset_sizes = get_data(
            data_dir, label_idx, args.batch_size,
            data_aug=False,
            color_jitter=args.color_jitter,
            jitter_brightness=args.jitter_brightness,
            jitter_contrast=args.jitter_contrast,
        )

    print(dataset_sizes)
    
    # Initialize model
    if args.pretrained_weights == 'random':
        print('Initialize ResNet18 without pretrained weights')
        model = initialize_model(num_classes, False)
    elif args.pretrained_weights == 'simclr':
        print('Initialize ResNet18 using SimCLR weights')
        print(f'SimCLR model path {args.simclr_path}')
        model = initialize_simclr_model(num_classes, args.simclr_path)
    elif args.pretrained_weights == 'auto_encoder':
        raise NotImplementedError()
    elif args.pretrained_weights == 'imagenet':
        print('Initialize ResNet18 using ImageNet pretrained weights')
        model = initialize_model(num_classes, True)
    else:
        raise NotImplementedError()

    # print(model)
    criterion = nn.CrossEntropyLoss()
    
    if args.weight_decay is not None:
        print(f"Set weight decay to {args.weight_decay}")
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr)
    
    lr_scheduler_ = lr_scheduler.LinearLR(
        optimizer_, 
        start_factor=1.0,
        end_factor=0.0,
        total_iters=args.epoch
    )
    
    # Train model
    print(f'===== Train Model with {args.portion}% data =====')
    print('-' * 50)
    print(f'{"Batch Size":<20}: {args.batch_size}')
    print(f'{"Learning Rate":<20}: {args.lr}')
    print(f'{"Weight Decay":<20}: {args.weight_decay if args.weight_decay else "None"}')
    print(f'{"Color Jitter":<20}: {args.color_jitter}')
    if args.color_jitter:
        print(f'{"  brightness":<20}: {args.jitter_brightness}')
        print(f'{"  contrast":<20}: {args.jitter_contrast}')
    print(f'{"Total Samples":<20}: {len(label_idx)}')
    print(f'{"Batches per Epoch":<20}: {len(label_idx) // args.batch_size}')

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{"Trainable Params":<20}: {total_trainable_params / 1e6:.2f}M')
    print('-' * 50)

    _, final_acc, _ = train_model(
        model, args.device, data_loaders, dataset_sizes,
        criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch
    )
    
    final_acc = round(final_acc, 4)
    print(f"Final Acc: {final_acc}")
    
    # Optional save model
    if args.ask_saving:
        user_input = input('Do you want to save the model checkpoint? (y/n): ').strip().lower()
        if user_input in ['y', 'yes']:
            model_path = input('Please enter the path to save the model: ').strip()
            if not model_path:
                print('No path provided. Using default: ./model_checkpoint.pth')
                model_path = './model_checkpoint.pth'
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            try:
                torch.save(model.state_dict(), model_path)            
                print(f'Model checkpoint saved to: {model_path}')
            except Exception as e:
                print(f'Error saving model: {e}')
        else:
            print('Model checkpoint not saved.')
    
    # Save results
    save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", f"cold_start_{args.pretrained_weights}")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    print(f'Aug config key: {aug_key}')
    
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
    
    if aug_key not in data:
        data[aug_key] = {}
    
    portion_key = str(float(args.portion))
    lr_key = str(args.lr)
    
    if portion_key not in data[aug_key]:
        data[aug_key][portion_key] = {}
    
    if lr_key not in data[aug_key][portion_key]:
        data[aug_key][portion_key][lr_key] = []
    
    data[aug_key][portion_key][lr_key].append(final_acc)
    
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
    
    save_compact_json(sorted_data, file_path)
    print(f'Result saved to {file_path}!')


if __name__ == "__main__":
    main()

# python3 run_first_iter_new.py --task_type hard --portion 20 --seed 42 --pretrained_weights imagenet --color_jitter --no_data_aug --lr 5e-4 --> 0.6941，有比較好!

# baseline 0.7176
# python3 run_first_iter_new.py --task_type hard --portion 20 --seed 42 --pretrained_weights imagenet --color_jitter --lr 3e-4 --> 0.7412，有比較好!
# python3 run_first_iter_new.py --task_type hard --portion 20 --seed 42 --pretrained_weights imagenet --color_jitter --lr 3e-4 --jitter_brightness 0.5 --jitter_contrast 0.5 --> 0.7059 
# python3 run_first_iter_new.py --task_type hard --portion 20 --seed 42 --pretrained_weights imagenet --color_jitter --lr 3e-4 --jitter_brightness 0.2 --jitter_contrast 0.2 --> 0.7294