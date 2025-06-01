import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, save_dir='checkpoints', verbose=False):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': []
        }
        self.best_acc = 0.0
        self.start_time = None
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 创建进度条
        pbar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1:3d}/{total_epochs} [Train]',
            leave=False,  # 不保留进度条
            ncols=120,
            file=sys.stdout,
            dynamic_ncols=True
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            current_acc = 100. * correct / total
            current_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        return avg_loss, acc
    
    def test(self, test_loader, criterion, epoch, total_epochs):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 创建进度条
        pbar = tqdm(
            test_loader, 
            desc=f'Epoch {epoch+1:3d}/{total_epochs} [Test ]',
            leave=False,  # 不保留进度条
            ncols=120,
            file=sys.stdout,
            dynamic_ncols=True
        )
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                current_acc = 100. * correct / total
                current_loss = total_loss / (batch_idx + 1)
                
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        avg_loss = total_loss / len(test_loader)
        acc = 100. * correct / total
        return avg_loss, acc
    
    def print_epoch_summary(self, epoch, total_epochs, train_loss, train_acc, 
                          test_loss, test_acc, epoch_time, lr, is_best=False):
        """打印epoch总结, 使用固定位置刷新"""
        elapsed_time = time.time() - self.start_time
        eta = (total_epochs - epoch - 1) * (elapsed_time / (epoch + 1))
        
        # 清除当前行并回到行首
        print('\r' + ' ' * 120 + '\r', end='')
        
        # 构建状态行
        status_line = (
            f"Epoch {epoch+1:3d}/{total_epochs} | "
            f"Train: {train_loss:.4f}/{train_acc:5.2f}% | "
            f"Test: {test_loss:.4f}/{test_acc:5.2f}% | "
            f"Best: {self.best_acc:5.2f}% | "
            f"Time: {epoch_time:4.1f}s | "
            f"ETA: {eta/60:4.0f}m | "
            f"LR: {lr:.1e}"
        )
        
        if is_best:
            status_line += " 🎉"
        
        print(status_line, flush=True)
    
    def fit(self, train_loader, test_loader, epochs=200, lr=0.1, weight_decay=5e-4, optimizer_type="sgd"):
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
        else:  # 默认使用 SGD
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            print(f"Optimizer: SGD(lr={lr}, momentum=0.9, weight_decay={weight_decay})")
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"Scheduler: CosineAnnealingLR(T_max={epochs})")
        print(f"Loss function: CrossEntropyLoss")
        print()
        
        self.start_time = time.time()
        
        # 打印表头
        print("=" * 120)
        print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Test Loss':>9} {'Test Acc':>8} | {'Best Acc':>8} | {'Time':>6} | {'ETA':>5} | {'LR':>8}")
        print("-" * 120)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练和测试
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, epoch, epochs
            )
            test_loss, test_acc = self.test(
                test_loader, criterion, epoch, epochs
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            # 保存最佳模型
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
                self.save_checkpoint(epoch, 'best')
            
            # 计算时间
            epoch_time = time.time() - epoch_start
            
            # 打印epoch总结（刷新显示）
            self.print_epoch_summary(
                epoch, epochs, train_loss, train_acc, 
                test_loss, test_acc, epoch_time, 
                optimizer.param_groups[0]['lr'], is_best
            )
            
            # 每10个epoch或最佳结果时显示详细信息
            if epoch % 10 == 0 or is_best or epoch == epochs - 1:
                if is_best:
                    print(f"    🎉 New best accuracy: {self.best_acc:.2f}% (saved checkpoint)")
        
        print("=" * 120)
        
        total_time = time.time() - self.start_time
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best test accuracy: {self.best_acc:.2f}%')
        
        return self.history
    
    def save_checkpoint(self, epoch, name='checkpoint'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / f'{name}.pth')
    
    def plot_curves(self, save_path='training_curves.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epochs, self.history['test_loss'], 'r-', label='Test Loss', linewidth=1.5)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=1.5)
        ax2.plot(epochs, self.history['test_acc'], 'r-', label='Test Acc', linewidth=1.5)
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Training curves saved to: {save_path}')