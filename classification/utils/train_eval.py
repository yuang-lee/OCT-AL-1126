import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import copy

def eval_model(model, device, data_loader, dataset_size, criterion):
    model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc  = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc.item()


def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler,
                num_epochs=20, history_out=None):
    """history_out：若傳入 dict，記錄學習曲線（不傳則零額外開銷、不影響原行為）：
      history_out["train_steps"] = [{step, epoch, train_loss}, ...]  ← 每個 training step(batch) 的 loss
      history_out["epochs"]      = [{epoch, train_loss, train_acc, val_loss, val_acc,
                                     test_loss, test_acc, lr}, ...]
                                   ← 每個 epoch 末的 val/test；**epoch 0 = 第一個訓練 epoch 開始前**的初始狀態
                                     （val loss、val acc、test acc 都在訓練前先記一筆）。"""
    model.to(device)
    since = time.time()

    # 初始化 best model 相關變量
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0

    # 學習曲線：epoch 0 = 訓練前先評估（第一個 epoch 開始前就記錄 loss 與 acc）
    global_step = 0
    if history_out is not None:
        history_out.setdefault("train_steps", [])
        history_out.setdefault("epochs", [])
        tr_l, tr_a = eval_model(model, device, data_loaders['train'], dataset_sizes['train'], criterion)
        va_l, va_a = eval_model(model, device, data_loaders['val'], dataset_sizes['val'], criterion)
        te_l, te_a = eval_model(model, device, data_loaders['test'], dataset_sizes['test'], criterion)
        history_out["epochs"].append({"epoch": 0, "train_loss": tr_l, "train_acc": tr_a,
                                      "val_loss": va_l, "val_acc": va_a,
                                      "test_loss": te_l, "test_acc": te_a,
                                      "lr": optimizer.param_groups[0]['lr']})
        history_out["train_steps"].append({"step": 0, "epoch": 0, "train_loss": tr_l})
        print(f'[epoch 0 / 訓練前] train_loss={tr_l:.4f} acc={tr_a:.4f} | '
              f'val_loss={va_l:.4f} acc={va_a:.4f} | test_acc={te_a:.4f}')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ep_rec = {}   # 暫存本 epoch 各 phase 的 (loss, acc) 給 history
        for phase in ['train', 'val']:
            if phase == 'train':
                print('train...')
                model.train()
            else:
                print('validate...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 每個 training step 記一筆 train loss
                if phase == 'train' and history_out is not None:
                    global_step += 1
                    history_out["train_steps"].append(
                        {"step": global_step, "epoch": epoch + 1, "train_loss": float(loss.item())})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            ep_rec[phase] = (epoch_loss, float(epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # 在 train phase 後打印當前 learning rate
            if phase == 'train':
                current_lr = optimizer.param_groups[0]['lr']
                print('Learning Rate: {:.6f}'.format(current_lr))

            # 保存 val loss 最低的模型
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('*** New best model found! (Val Loss: {:.4f}) ***'.format(best_val_loss))

        # 記本 epoch 末學習曲線（epoch 從 1 起算，接在 epoch 0=訓練前 之後）；另評估 test
        if history_out is not None and 'train' in ep_rec and 'val' in ep_rec:
            te_l, te_a = eval_model(model, device, data_loaders['test'], dataset_sizes['test'], criterion)
            history_out["epochs"].append({"epoch": epoch + 1,
                                          "train_loss": ep_rec['train'][0], "train_acc": ep_rec['train'][1],
                                          "val_loss": ep_rec['val'][0], "val_acc": ep_rec['val'][1],
                                          "test_loss": te_l, "test_acc": te_a,
                                          "lr": optimizer.param_groups[0]['lr']})

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f} at epoch {}'.format(best_val_loss, best_epoch))
    
    # 載入最佳模型
    model.load_state_dict(best_model_wts)
    
    # 在 test set 上評估
    print('\n' + '='*50)
    print('Evaluating best model on test set...')
    print('='*50)
    test_loss, test_acc = eval_model(
        model, device, data_loaders['test'], dataset_sizes['test'], criterion
    )
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

    # 第三個回傳值 best_val_loss：供 AL 以 val（非 test）挑最佳 lr，避免 test leakage
    return model, test_acc, best_val_loss
