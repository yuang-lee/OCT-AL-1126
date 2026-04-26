## 0401 final

### Training from random init
不知為啥，跟from ImageNet不同，
from random 的話，似乎反而是Portion越少的時候需要越小的LR!
但這不影響啦

```bash
cd ./classification

AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="7e-6 1e-5 3e-5 5e-5 7e-5 1e-4 3e-4" PORTIONS="2.5 5 10" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="5e-5 7e-5 1e-4 3e-4 5e-4" PORTIONS="20 30 40" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=3 SEEDS="10 24 38 42 57" LRS="3e-4 5e-4 7e-4 1e-3 " PORTIONS="50 60 70" DEVICE="cuda:4" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=4 SEEDS="10 24 38 42 57" LRS="5e-4 7e-4 1e-3 3e-3" PORTIONS="80 90" DEVICE="cuda:2" ./exp/meta_scripts/train_parallel.sh

AUGS="aug4" MAX_RUN=4 SEEDS="42" LRS="7e-4 1e-3 3e-3" PORTIONS="100" DEVICE="cuda:2" ./exp/meta_scripts/train_parallel.sh
```

### Training from ImageNet
Already done in data_aug

### Training from SimCLR1
#### 1. train SimCLR models
Remeber to do SimCLR varying batch size (16, 32, 64, 128, 256) and epoch (10, 20, 50, 100, 200, 500) for 100% data fine-tuning first!
```bash
cd ../ # i.e., root dir of this project
./classification/exp/weights_init/scripts/train_simclr.sh --DEVICE 'cuda:0'
./classification/exp/weights_init/scripts/train_simclr_2.sh --DEVICE 'cuda:1'
./classification/exp/weights_init/scripts/train_simclr_3.sh --DEVICE 'cuda:2'

# view run
tensorboard --logdir ./SSL/simclr/tb_logs
```

#### 2. fine-tune on 100% OCT data
```bash
cd ./classification
./exp/weights_init/scripts/simclr_ft_100_4.sh # 5e-5
./exp/weights_init/scripts/simclr_ft_100_3.sh # 7e-5
./exp/weights_init/scripts/simclr_ft_100_1.sh # 1e-4
./exp/weights_init/scripts/simclr_ft_100_2.sh # 3e-4


# 最好也來個ft 10%的? 之後可以做成兩個separate heatmaps?
# 橫軸是epoch, 縱軸是bs
./exp/weights_init/scripts/simclr_ft_10_5.sh # 5e-5
./exp/weights_init/scripts/simclr_ft_10_2.sh # 7e-5 
./exp/weights_init/scripts/simclr_ft_10_3.sh # 1e-4
./exp/weights_init/scripts/simclr_ft_10_1.sh # 3e-4
./exp/weights_init/scripts/simclr_ft_10_4.sh # 5e-4
./exp/weights_init/scripts/simclr_ft_10_6.sh # 7e-4


## bs=256 單獨RUN一下
AUGS="aug4" MAX_RUN=3 SEEDS="42" LRS="1e-4 3e-4 5e-4" PORTIONS="100" DEVICE="cuda:4" PRETRAINED="simclr" RUNS=1 \
SIMCLR_BS="256" \
SIMCLR_EP="500" \
./exp/meta_scripts/train_parallel_simclr.sh

AUGS="aug4" MAX_RUN=3 SEEDS="42" LRS="3e-5 5e-5 7e-5" PORTIONS="100" DEVICE="cuda:3" PRETRAINED="simclr" RUNS=1 \
SIMCLR_BS="256" \
SIMCLR_EP="500" \
./exp/meta_scripts/train_parallel_simclr.sh



## 最後最好用:

### portions = 100
SIMCLR_EPS="10 20 50" \
SIMCLR_BSS="16 32 64" \
AUGS="aug4" \
MAX_RUN=5 \
RUNS=3 \
SEEDS="42" \
LRS="5e-5" \
PORTIONS="100" \
DEVICE="cuda:0" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh

SIMCLR_EPS="100 200 500" \
SIMCLR_BSS="128 256" \
AUGS="aug4" \
MAX_RUN=5 \
RUNS=3 \
SEEDS="42" \
LRS="5e-5" \
PORTIONS="100" \
DEVICE="cuda:1" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh

SIMCLR_EPS="10 20 50" \
SIMCLR_BSS="16 32 64" \
AUGS="aug4" \
MAX_RUN=5 \
RUNS=3 \
SEEDS="42" \
LRS="1e-4" \
PORTIONS="100" \
DEVICE="cuda:2" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh

SIMCLR_EPS="100 200 500" \
SIMCLR_BSS="128 256" \
AUGS="aug4" \
MAX_RUN=5 \
RUNS=3 \
SEEDS="42" \
LRS="1e-4" \
PORTIONS="100" \
DEVICE="cuda:3" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh


### portions = 10

SIMCLR_EPS="10 20 50 100 200 500" \
SIMCLR_BSS="16 32 64 128 256" \
AUGS="aug4" \
MAX_RUN=3 \
RUNS=3 \
SEEDS="10 24" \
LRS="1e-4 5e-4" \
PORTIONS="10" \
DEVICE="cuda:6" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh

SIMCLR_EPS="10 20 50 100 200 500" \
SIMCLR_BSS="16 32 64 128 256" \
AUGS="aug4" \
MAX_RUN=3 \
RUNS=3 \
SEEDS="38 42" \
LRS="1e-4 5e-4" \
PORTIONS="10" \
DEVICE="cuda:7" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh


SIMCLR_EPS="10 20 50 100 200 500" \
SIMCLR_BSS="16 32 64 128 256" \
AUGS="aug4" \
MAX_RUN=3 \
RUNS=3 \
SEEDS="57" \
LRS="1e-4 5e-4" \
PORTIONS="10" \
DEVICE="cuda:8" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh


SIMCLR_EPS="10 20 50 100 200 500" \
SIMCLR_BSS="256" \
AUGS="aug4" \
MAX_RUN=3 \
RUNS=3 \
SEEDS="10 24 38 42 57" \
LRS="1e-4 5e-4" \
PORTIONS="10" \
DEVICE="cuda:5" \
PRETRAINED="simclr" \
./exp/weights_init/scripts/simclr_meta.sh


```


Visualize
```bash
python3 plot_simclr_heatmap.py \
  --portion 10 \
  --simclr_lr 0.0002 \
  --epochs 10 20 50 100 200 500 \
  --batch_sizes 16 32 64 128 256 \
  --save_dir ./

python3 plot_simclr_heatmap.py \
  --portion 100 \
  --simclr_lr 0.0002 \
  --epochs 10 20 50 100 200 500 \
  --batch_sizes 16 32 64 128 256 \
  --save_dir ./

```

### Training from SimCLR2
Remeber to do SimCLR varying batch size and epoch for 100% data fine-tuning first!

Remember to save in different model ckpt names!

### visualize
```bash
cd ./classification/exp/weights_init
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




## Plot tSNE

```bash
python3 ./classification/exp/weights_init/plot_tsne.py --task_type hard --pretrained_weights random
python3 ./classification/exp/weights_init/plot_tsne.py --task_type hard --pretrained_weights imagenet


python3 ./classification/exp/weights_init/plot_tsne.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl

# 先不要用有vertical flip的simclr，因為vertical沒那麼多不同epochs...
python3 ./classification/exp/weights_init/plot_tsne.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep10.pkl


python3 ./classification/exp/weights_init/plot_tsne.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep100.pkl
```

## Linear eval
### plot
```bash
python3 ./classification/exp/weights_init/plot_pretrain_effect.py


```

### Random
```bash
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights random --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.4449
#   Test Accuracy: 0.4784

python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights random --seed 42 --portion 100
# Final Results:
#   Validation Accuracy: 0.5000
#   Test Accuracy: 0.4824
```

### ImageNet
```bash
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights imagenet --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5276
#   Test Accuracy: 0.4863

python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights imagenet --seed 42 --portion 100
# Final Results:
#   Validation Accuracy: 0.7756
#   Test Accuracy: 0.8039
```

### SimCLR
```bash
### w/ Vertical
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5827
#   Test Accuracy: 0.5922

### w/o Vertical 
#### ep=10
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep10.pkl --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5472
#   Test Accuracy: 0.5765
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep10.pkl --seed 42 --portion 100
# Final Results:
#   Validation Accuracy: 0.8346
#   Test Accuracy: 0.7843

#### ep=25
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep25.pkl --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5591
#   Test Accuracy: 0.5686

#### ep=100
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep100.pkl --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5512
#   Test Accuracy: 0.5882
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep100.pkl --seed 42 --portion 100
# Final Results:
#   Validation Accuracy: 0.8268
#   Test Accuracy: 0.8039


#### ep=300
python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl --seed 42 --portion 5
# Final Results:
#   Validation Accuracy: 0.5669
#   Test Accuracy: 0.5922

python3 ./classification/exp/weights_init/linear_eval.py --task_type hard --pretrained_weights simclr --simclr_path ./classification/model/simclr/ckpt_wo_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl --seed 42 --portion 100
# Final Results:
#   Validation Accuracy: 0.8189
#   Test Accuracy: 0.8196
```


## Examine different SimCLR pretraining on downstream performance
### Run pretraining:
Note that the learning rate is linearly scaled with batch size

```bash
## Start from ImageNet pretrained ckpt!!
python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 128 --lr 2e-4 --gpu-index 1 -data ./ds/classification/seven_class/train

python3 ./SSL/simclr/run.py --arch resnet18 --epochs 100 --batch-size 256 --lr 4e-4 --gpu-index 1 -data ./ds/classification/seven_class/train

## 12/2 include vertical flip data augmentation!
python3 ./SSL/simclr/run.py --arch resnet18 --epochs 300 --batch-size 256 --lr 4e-4 --gpu-index 1 -data ./ds/classification/seven_class/train
```

### Plot
```bash
python3 ./classification/figs/weights_init/plot_simclr.py
```



## See whether the performance from SimCLR pretraining can scale to more label portion
### Run
```bash
./classification/scripts/for_loop_train_one_random.sh
./classification/scripts/for_loop_train_one_imagenet.sh
./classification/scripts/for_loop_train_one_simclr.sh
```

### Plot
```bash
python3 ./classification/exp/weights_init/plot_across_label_portion.py
```


### Random 
#### 5% w Random Initialization
```bash
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --no_use_pretrained

[0.3922, 0.4157, 0.3176]
```

#### 5% w ImageNet Pretraining
```bash
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42
[0.5333, 0.5529, 0.5608]
```

#### 5% w SimCLR pretraining
```bash
# bs=32

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr5e-05_bs32_ep10.pkl' \
        --exp_path './exp_results_temp' \
        --no_data_aug
[0.5608]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr5e-05_bs32_ep10.pkl'
[0.6000]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr5e-05_bs32_ep25.pkl'
[0.5843, 0.5804, 0.5961]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr5e-05_bs32_ep50.pkl'
[0.5922, 0.5765, 0.5882]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr5e-05_bs32_ep100.pkl'
[0.6000, 0.5765, 0.5961]

# bs=64
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep10.pkl'
[0.5882, 0.5686, 0.5686]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep25.pkl'
[0.5725, 0.5961, 0.6157]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep50.pkl'
[0.6235, 0.5843, 0.6196]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep100.pkl' \
        --exp_path './exp_results_temp'

[0.6157]
# [0.6745, 0.6431, 0.6549]


#bs=128
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep10.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep25.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep50.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl' \
        --exp_path './exp_results_temp'

[0.6039]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep300.pkl' \
        --exp_path './exp_results_temp'


#bs=256
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep10.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep25.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep50.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep100.pkl' \
        --exp_path './exp_results_temp'

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'

[0.5490] # 居然真的下降了...

# 如果不用data aug會不會改善? 還真的改善??!!
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --no_data_aug \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'
[0.60000]

# 如果只用horizontal flip data aug?
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal' \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'

[0.5922]

# 如果只用vertical flip data aug? 確實跟只有horizontal flip比起來很慘? Yes!
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical' \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'

[0.5647]

# 改simclr 也有 vertical aug
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:7' \
        --portion 5 \
        --seed 42 \
        --no_data_aug \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'
[0.5765]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:7' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'horizontal' \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'

[0.5725]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:7' \
        --portion 5 \
        --seed 42 \
        --aug_factor 2 \
        --flip_type 'vertical' \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl' \
        --exp_path './exp_results_temp'

[0.5686]
```


#### 10% w SimCLR pretraining
```bash
#bs=128
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep10.pkl'
[0.6275, 0.6039, 0.6510]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep25.pkl'
[0.6745, 0.6588, 0.6588]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep50.pkl'
[0.6667, 0.6588, 0.6549]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
[0.6706, 0.6431, 0.6627]


#bs=256
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep10.pkl'
[0.6471, 0.6549, 0.6392]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep25.pkl'
[67.84, 65.10, 65.88]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:2' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep50.pkl'
[0.6353, 0.6627, 0.6784]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep100.pkl'
[0.6824, 0.6471, 0.6706]

```


#### 10%
```bash
# bs=64
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep100.pkl'
# [0.597250]
[0.6510]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0001_bs64_ep300.pkl'
# [0.669941]
[0.6510]

# bs=128
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
# [0.654224]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0002_bs128_ep300.pkl'
# [0.654224]

# bs=256
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep100.pkl'
# [0.640472]
[0.6510]

python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:1' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0004_bs256_ep300.pkl'
# [0.669941]
[0.6471]

# bs=512
python3 ./classification/run_first_iter.py \
        --task_type 'hard' \
        --device 'cuda:0' \
        --portion 10 \
        --seed 42 \
        --pretrained_weights 'simclr' \
        --simclr_path './classification/model/simclr/resnet18_simclr_lr0.0008_bs512_ep100.pkl'
# [0.648330]

```

