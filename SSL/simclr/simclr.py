import logging
import os
import sys
import json
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        # self.writer = SummaryWriter(log_dir='./SSL/simclr/tb_logs')
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir = os.path.join("./SSL/simclr/tb_logs", time_str)
        # self.writer = SummaryWriter(log_dir=log_dir)
        # logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
        #                     level=logging.DEBUG)
        # 建立包含更多資訊的log目錄名稱
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{self.args.arch}_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}_{time_str}"
        log_dir = os.path.join("./SSL/simclr/tb_logs", exp_name)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.DEBUG
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        # 建立資料夾
        model_dir = "./SSL/simclr/ckpt"
        json_dir  = "./SSL/simclr/json"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(json_dir,  exist_ok=True)

        # 共用檔名 (不含副檔名)
        base_name = f"{self.args.arch}_simclr_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}"
        model_path = os.path.join(model_dir, f"{base_name}.pkl")
        json_path  = os.path.join(json_dir,  f"{base_name}.json")

        # JSON 結構初始化
        training_history = {
            "arch":       self.args.arch,
            "lr":         self.args.lr,
            "batch_size": self.args.batch_size,
            "epochs":     self.args.epochs,
            "history":    []   # 每個 epoch append 一筆
        }

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            epoch_loss = 0.0
            epoch_top1 = 0.0
            epoch_top5 = 0.0
            batch_count = 0

            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                epoch_loss += loss.item()
                epoch_top1 += top1[0].item()
                epoch_top5 += top5[0].item()
                batch_count += 1
                n_iter += 1

            avg_loss    = epoch_loss / batch_count
            avg_top1    = epoch_top1 / batch_count
            avg_top5    = epoch_top5 / batch_count
            current_lr  = self.scheduler.get_lr()[0]

            # TensorBoard
            self.writer.add_scalar('epoch_loss',      avg_loss,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top1',  avg_top1,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top5',  avg_top5,   global_step=epoch_counter)
            self.writer.add_scalar('learning_rate',   current_lr, global_step=epoch_counter)

            # Terminal
            print(f"Epoch: {epoch_counter}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Top1 Accuracy: {avg_top1:.2f}%")
            print(f"  Top5 Accuracy: {avg_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print("-" * 50)

            # JSON 動態更新 ← 每個 epoch 都寫入一次
            training_history["history"].append({
                "epoch":      epoch_counter,
                "loss":       avg_loss,
                "top1":       avg_top1,
                "top5":       avg_top5,
                "lr":         current_lr,
            })
            with open(json_path, "w") as f:
                json.dump(training_history, f, indent=4)

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}")

        # 存 model
        torch.save(self.model.state_dict(), model_path)
        logging.info("Training has finished.")

        
    # def train(self, train_loader):

    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         epoch_loss = 0.0
    #         epoch_top1 = 0.0
    #         epoch_top5 = 0.0
    #         batch_count = 0
            
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)
    #             images = images.to(self.args.device)

    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)
                
    #             self.optimizer.zero_grad()
    #             scaler.scale(loss).backward()
    #             scaler.step(self.optimizer)
    #             scaler.update()

    #             # 累積每個 epoch 的統計資訊
    #             top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #             epoch_loss += loss.item()
    #             epoch_top1 += top1[0].item()
    #             epoch_top5 += top5[0].item()
    #             batch_count += 1

    #             n_iter += 1
            
    #         # 計算平均值
    #         avg_loss = epoch_loss / batch_count
    #         avg_top1 = epoch_top1 / batch_count
    #         avg_top5 = epoch_top5 / batch_count
    #         current_lr = self.scheduler.get_lr()[0]
            
    #         # 記錄到 tensorboard 並同時 print 到 terminal
    #         self.writer.add_scalar('epoch_loss', avg_loss, global_step=epoch_counter)
    #         self.writer.add_scalar('epoch_acc/top1', avg_top1, global_step=epoch_counter)
    #         self.writer.add_scalar('epoch_acc/top5', avg_top5, global_step=epoch_counter)
    #         self.writer.add_scalar('learning_rate', current_lr, global_step=epoch_counter)
            
    #         # Print 到 terminal
    #         print(f"Epoch: {epoch_counter}")
    #         print(f"  Loss: {avg_loss:.4f}")
    #         print(f"  Top1 Accuracy: {avg_top1:.2f}%")
    #         print(f"  Top5 Accuracy: {avg_top5:.2f}%")
    #         print(f"  Learning Rate: {current_lr:.6f}")
    #         print("-" * 50)
            
    #         # # warmup for the first 10 epochs
    #         # if epoch_counter >= 10:
            
    #         ## no warmup, directly consine decay
    #         self.scheduler.step()
        
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}")
            
    #     torch.save(self.model.state_dict() ,f"./SSL/simclr/{self.args.arch}_simclr_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}.pkl")
    #     logging.info("Training has finished.")

    # def train(self, train_loader):

    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)

    #             images = images.to(self.args.device)

    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)
    #             self.optimizer.zero_grad()

    #             scaler.scale(loss).backward()

    #             scaler.step(self.optimizer)
    #             scaler.update()
    #             if n_iter % self.args.log_every_n_steps == 0:
    #                 top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #                 self.writer.add_scalar('loss', loss, global_step=n_iter)
    #                 self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
    #                 self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
    #                 self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

    #             n_iter += 1
    #         print("epoch: ",epoch_counter, " loss: ", loss.item())
    #         # warmup for the first 10 epochs
    #         if epoch_counter >= 10:
    #             self.scheduler.step()
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
    #     torch.save(self.model.state_dict() ,f"./SSL/simclr/{self.args.arch}_simclr_lr{self.args.lr}_ep{self.args.epochs}.pkl")
    #     logging.info("Training has finished.")


    # def train_with_sam(self, train_loader):
    #     """
    #     SimCLR training function with Sharpness Aware Minimization (SAM) support
    #     """
    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training with SAM for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)
    #             images = images.to(self.args.device)

    #             # Zero gradients
    #             self.optimizer.zero_grad()

    #             # First forward pass and backward pass for SAM
    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)

    #             scaler.scale(loss).backward()

    #             # Define closure function for SAM
    #             def closure():
    #                 self.optimizer.zero_grad()
    #                 with autocast(enabled=self.args.fp16_precision):
    #                     features = self.model(images)
    #                     logits, labels = self.info_nce_loss(features)
    #                     loss = self.criterion(logits, labels)
    #                 scaler.scale(loss).backward()
    #                 return loss

    #             # SAM optimization step
    #             scaler.step(self.optimizer, closure)
    #             scaler.update()

    #             # Logging
    #             if n_iter % self.args.log_every_n_steps == 0:
    #                 top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #                 self.writer.add_scalar('loss', loss, global_step=n_iter)
    #                 self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
    #                 self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
    #                 self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

    #             n_iter += 1

    #         print("epoch: ", epoch_counter, " loss: ", loss.item())
            
    #         # warmup for the first 10 epochs
    #         if epoch_counter >= 10:
    #             self.scheduler.step()
            
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    #     torch.save(self.model.state_dict(), f"./SSL/simclr/{self.args.arch}_simclr_sam_lr{self.args.lr}_ep{self.args.epochs}.pkl")
    #     logging.info("Training with SAM has finished.")