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


def train_model(model, device, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    model.to(device)
    since = time.time()
    
    # 初始化 best model 相關變量
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
