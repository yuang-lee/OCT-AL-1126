# 0323

## train flipping
### individual
```bash
cd ./classification
python3 ./run_first_iter.py \
                --task_type 'hard' --lr 5e-4 \
                --pretrained_weights imagenet \
                --device 'cuda:7' \
                --portion 10 --seed 24 \
                --no_data_aug

python3 ./run_first_iter.py \
                --task_type 'hard' --lr 1e-4 \
                --pretrained_weights imagenet \
                --device 'cuda:7' \
                --portion 100 --seed 42

python3 ./run_first_iter.py \
                --task_type 'hard' --lr 1e-4 \
                --pretrained_weights imagenet \
                --device 'cuda:8' \
                --aug_factor 2 \
                --flip_type vertical \
                --portion 100 --seed 42
```

### use script
```bash
cd ./classification
## portions = [2.5 5 10 20 40 80]
## seeds = [10, 24, 38, 42, 57]
# SEED=10 PORTIONS="2.5 5 10 20 40 80" DEVICE="cuda:0" ./exp/data_aug/scripts/train_flip.sh
SEED=10 PORTIONS="2.5" DEVICE="cuda:0" ./exp/data_aug/scripts/train_flip.sh
SEED=24 PORTIONS="2.5" DEVICE="cuda:3" ./exp/data_aug/scripts/train_flip.sh
SEED=38 PORTIONS="2.5" DEVICE="cuda:1" ./exp/data_aug/scripts/train_flip.sh
SEED=42 PORTIONS="2.5" DEVICE="cuda:3" ./exp/data_aug/scripts/train_flip.sh
SEED=57 PORTIONS="2.5" DEVICE="cuda:2" ./exp/data_aug/scripts/train_flip.sh

SEED=24 PORTIONS="10" DEVICE="cuda:1" ./exp/data_aug/scripts/train_flip.sh
SEED=38 PORTIONS="10" DEVICE="cuda:2" ./exp/data_aug/scripts/train_flip.sh
SEED=57 PORTIONS="10" DEVICE="cuda:1" ./exp/data_aug/scripts/train_flip.sh


./exp/data_aug/scripts/flip_total_p20.sh
./exp/data_aug/scripts/flip_total_p40.sh
```


## 以下是最後確定最好用的腳本(.sh)!
```bash
cd ./classification

AUGS="no_aug aug2_h aug2_v" MAX_RUN=3 SEEDS="24 38 57" LRS="7e-5 1e-4 3e-4 5e-4 7e-4" PORTIONS="60" DEVICE="cuda:7" ./exp/data_aug/scripts/train_flip_parallel.sh

AUGS="no_aug aug2_h aug2_v aug3 aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="5e-5 7e-5 1e-4 3e-4 5e-4" PORTIONS="10 20 30 40" DEVICE="cuda:7" ./exp/data_aug/scripts/train_flip_parallel.sh


AUGS="no_aug" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="3e-4 5e-4 7e-4" PORTIONS="70 80 90" DEVICE="cuda:8" ./exp/data_aug/scripts/train_flip_parallel.sh


AUGS="aug2_h aug2_v" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="1e-4 3e-4" PORTIONS="70" DEVICE="cuda:8" ./exp/data_aug/scripts/train_flip_parallel.sh

## 4/26 最後補充
### portion 90, horizontal
AUGS="aug2_h" MAX_RUN=3 SEEDS="10 24" LRS="1e-4" PORTIONS="90" DEVICE="cuda:0" ./exp/data_aug/scripts/train_flip_parallel.sh
AUGS="aug2_h" MAX_RUN=3 SEEDS="38 57" LRS="3e-4" PORTIONS="90" DEVICE="cuda:1" ./exp/data_aug/scripts/train_flip_parallel.sh
AUGS="aug2_h" MAX_RUN=3 SEEDS="42" LRS="7e-4" PORTIONS="90" DEVICE="cuda:2" ./exp/data_aug/scripts/train_flip_parallel.sh

### portion 90, vertical
AUGS="aug2_v" MAX_RUN=3 SEEDS="10 38 42" LRS="3e-4" PORTIONS="90" DEVICE="cuda:7" ./exp/data_aug/scripts/train_flip_parallel.sh
AUGS="aug2_v" MAX_RUN=3 SEEDS="24 57" LRS="1e-4" PORTIONS="90" DEVICE="cuda:8" ./exp/data_aug/scripts/train_flip_parallel.sh

### portion 90, 3X
AUGS="aug3" MAX_RUN=3 SEEDS="10 57" LRS="7e-5" PORTIONS="90" DEVICE="cuda:0" ./exp/data_aug/scripts/train_flip_parallel.sh

AUGS="aug3" MAX_RUN=3 SEEDS="24" LRS="1e-4" PORTIONS="90" DEVICE="cuda:4" ./exp/data_aug/scripts/train_flip_parallel.sh

AUGS="aug3" MAX_RUN=3 SEEDS="38" LRS="5e-4" PORTIONS="90" DEVICE="cuda:5" ./exp/data_aug/scripts/train_flip_parallel.sh

AUGS="aug3" MAX_RUN=3 SEEDS="42" LRS="5e-5" PORTIONS="90" DEVICE="cuda:6" ./exp/data_aug/scripts/train_flip_parallel.sh


### portion 90, 4X
AUGS="aug4" MAX_RUN=3 SEEDS="10 38" LRS="7e-5" PORTIONS="90" DEVICE="cuda:2" ./exp/data_aug/scripts/train_flip_parallel.sh

```

## visualize
```bash
cd ./classification/exp/data_aug
python3 ./plot_all.py --portions 2.5
python3 ./plot_all.py --portions 5
python3 ./plot_all.py --portions 10
python3 ./plot_all.py --portions 20
python3 ./plot_all.py --portions 30
python3 ./plot_all.py --portions 40
python3 ./plot_all.py --portions 50
python3 ./plot_all.py --portions 60
python3 ./plot_all.py --portions 70
python3 ./plot_all.py --portions 80
python3 ./plot_all.py --portions 90
python3 ./plot_all.py --portions 100
```

Oh...LR應該還要再往下一格，尤其是data量大、又有4x augmentation的時候! --> fixed?




----下面是舊的----

### Run Scripts
```bash
./classification/exp/data_aug/scripts/portion_5_data_aug_random.sh
./classification/exp/data_aug/scripts/portion_5_data_aug_imagenet.sh

```



### Plot
```bash
python3 ./classification/exp/data_aug/plot_imagenet_portion5.py
```





### Random Init. (5%)
```bash
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 5 \
        --seed 42 \
        --no_use_pretrained \
        --no_data_aug
[0.4196, 0.3961]

# horizontal flip
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:6' \
        --portion 5 \
        --seed 42 \
        --no_use_pretrained \
        --aug_factor 2

# vertical flip
////

python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:6' \
        --portion 5 \
        --seed 42 \
        --no_use_pretrained \
        --aug_factor 4
[0.3922, 0.4157, 0.3176]

```
### ImageNet Pretrain (1%)
```bash
# no aug
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 1 \
        --seed 42 \
        --no_data_aug

[0.1490, 0.1882]

# 2x aug (horizontal)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 1 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal'
[14.51]

# 2x aug (vertical)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 1 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical'
[12.94]

# 4x aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 1 \
        --seed 42
[0.2157]
```
### ImageNet Pretrain (2.5%)
```bash
# no aug
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 2.5 \
        --seed 42 \
        --no_data_aug

[0.4431]

# 2x aug (horizontal)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 2.5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal'
[0.4157]

# 2x aug (vertical)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 2.5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical'
[0.3961] 

# 4x aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 2.5 \
        --seed 42
[0.4667]
```


### ImageNet Pretrain (5%)
```bash
# no aug
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --no_data_aug \
        --exp_path './exp_results_temp'


[0.4824, 0.5333, 0.4824, 0.4863, 0.5216, 0.5176, 0.4941, 0.5098, 0.4980, 0.4784]

# 2x aug (horizontal)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal' \
        --exp_path './exp_results_temp'

[0.5176, 0.5608, 0.5373, 0.5098, 0.5294, 0.5412, 0.5529, 0.5294, 0.5294, 0.5451]

# 2x aug (vertical)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical'  \
        --exp_path './exp_results_temp'

[0.5137, 0.5451, 0.5529, 0.5216, 0.5294, 0.5294, 0.5490, 0.4980, 0.5294, 0.5451]

# 3x aug (horizontal + vertical flip)
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:8' \
        --portion 5 \
        --seed 42 \
        --aug_factor 3 \
        --exp_path './exp_results_temp'
[0.5373, 0.5569, 0.6039, 0.5608, 0.5294, 0.5255, 0.5490, 0.5569, 0.5686, 0.5451]

# 4x aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 5 \
        --seed 42 \
        --exp_path './exp_results_temp'

[0.5333, 0.5529, 0.5608, 0.5647, 0.5451, 0.5569, 0.5294, 0.5608, 0.5647]
```


### ImageNet Pretrain (5%)
```bash
# no aug
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 100 \
        --seed 42 \
        --no_data_aug \
        --exp_path './exp_results_temp_no_aug'

# 2x aug (horizontal)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 100 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal' \
        --exp_path './exp_results_temp_aug2_horizontal'

# 2x aug (vertical)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 100 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical'  \
        --exp_path './exp_results_temp_aug2_vertical'

# 3x aug (horizontal + vertical flip)
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:8' \
        --portion 100 \
        --seed 42 \
        --aug_factor 3 \
        --exp_path './exp_results_temp_aug3'

# 4x aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 100 \
        --seed 42 \
        --exp_path './exp_results_temp_aug4'

```

### ImageNet Pretrain (10%)
```bash
# no aug
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 10 \
        --seed 42 \
        --no_data_aug \
        --exp_path './exp_results_temp'

[0.6039, 0.5882, 0.6235, 0.5961]

# 2x aug (horizontal)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 10 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal' \
        --exp_path './exp_results_temp'

[0.5961, 0.6235]

# 2x aug (vertical)
python3 ./classification/run_first_iter_new.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 10 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical' \
        --exp_path './exp_results_temp'

[0.6118, 0.6314, 0.6078, 0.6157]

# 4x aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:4' \
        --portion 10 \
        --seed 42 \
        --exp_path './exp_results_temp'

[0.6196, 0.6431]
```



還要做一個是SimCLR pretrain with and without data aug across different label portion!