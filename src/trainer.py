import os
import time
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .config import TRAIN_CONFIG, DATA_CONFIG, DEVICE
from .models.siamese_net import SiameseNet
from .models.loss import get_loss_function, normalize_params

# 配置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class Trainer:
    def __init__(self,
                model: SiameseNet,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                optimizer: Optimizer,
                scheduler: Optional[_LRScheduler] = None,
                loss_fn: Optional[Callable] = None,
                device: torch.device = DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or get_loss_function()
        self.device = device
        
        self.num_epochs = TRAIN_CONFIG['NUM_EPOCHS']
        self.save_freq = TRAIN_CONFIG['SAVE_FREQ']
        self.max_models = TRAIN_CONFIG['MAX_MODELS']
        self.save_best = TRAIN_CONFIG['SAVE_BEST']
        self.model_dir = DATA_CONFIG['MODEL_DIR']
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.saved_models = []
        
        self.train_losses = []
        self.val_losses = []
    
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", 
            leave=True,
            position=0,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            vis_imgs = batch['vis_img'].to(self.device)
            ir_imgs = batch['ir_img'].to(self.device)
            targets = batch['transform_params'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(ir_imgs, vis_imgs)
            
            loss = self.loss_fn(outputs, targets)
            
            loss.backward()
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(self.train_loader)
        
        return avg_loss
    
    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        
        progress_bar = tqdm(
            self.valid_loader, 
            desc="Validation", 
            leave=True,
            position=0,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                vis_imgs = batch['vis_img'].to(self.device)
                ir_imgs = batch['ir_img'].to(self.device)
                targets = batch['transform_params'].to(self.device)
                
                outputs = self.model(ir_imgs, vis_imgs)
                
                loss = self.loss_fn(outputs, targets)
                
                val_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(self.valid_loader)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> str:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        regular_path = os.path.join(self.model_dir, f"epoch_{epoch+1}_loss_{loss:.4f}.pth")
        
        torch.save(checkpoint, regular_path)
        
        self.saved_models.append(regular_path)
        
        last_path = os.path.join(self.model_dir, "last.pth")
        torch.save(checkpoint, last_path)
        
        if is_best:
            best_path = os.path.join(self.model_dir, "best.pth")
            torch.save(checkpoint, best_path)
        
        return regular_path
    
    def manage_saved_models(self):
        regular_models = [m for m in self.saved_models if not m.endswith(("best.pth", "last.pth"))]
        
        if len(regular_models) > self.max_models:
            models_to_remove = regular_models[:-self.max_models]
            for model_path in models_to_remove:
                if os.path.exists(model_path):
                    os.remove(model_path)
                self.saved_models.remove(model_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_loss = checkpoint['loss']
        next_epoch = checkpoint['epoch'] + 1
        
        return next_epoch
    
    def create_loss_summary_plot(self, train_losses: List[float], val_losses: List[float], save_path: str):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='训练损失')
        plt.plot(epochs, val_losses, 'r-', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epochs')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, resume_from: Optional[str] = None) -> Dict:
        if resume_from:
            print(f"从检查点恢复训练: {resume_from}")
            self.start_epoch = self.load_checkpoint(resume_from)
            print(f"从epoch {self.start_epoch} 继续训练")
        else:
            print("从头开始训练")
        
        print(f"训练集: {len(self.train_loader.dataset)} 个样本")
        print(f"验证集: {len(self.valid_loader.dataset)} 个样本")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            print(f"Epoch: {epoch+1:02}/{self.num_epochs} | 用时: {epoch_mins:.0f}m {epoch_secs:.0f}s")
            print(f"训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
            
            if is_best:
                print(f"*** 新的最佳验证损失: {val_loss:.6f} ***")
            
            should_save = ((epoch + 1) % self.save_freq == 0) or (epoch == self.num_epochs - 1)
            if should_save or (is_best and self.save_best):
                saved_path = self.save_checkpoint(epoch, val_loss, is_best)
                print(f"已保存检查点: {saved_path}")
                self.manage_saved_models()
            
            print()
            
            loss_plot_path = os.path.join(self.model_dir, 'loss_plot.png')
            self.create_loss_summary_plot(self.train_losses, self.val_losses, loss_plot_path)
        
        return {
            'epochs': self.num_epochs,
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'best_val_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        } 