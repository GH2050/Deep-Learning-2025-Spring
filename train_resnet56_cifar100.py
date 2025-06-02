#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
import json
import csv
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import argparse

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging(save_dir, rank=0):
    """设置日志记录"""
    if rank == 0:
        log_file = os.path.join(save_dir, 'detailed_training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("="*50)
        logger.info("ResNet-56 CIFAR-100 训练开始")
        logger.info("="*50)
        return logger
    return None

def log_system_info(logger, rank=0):
    """记录系统信息"""
    if rank == 0 and logger:
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前设备: {torch.cuda.current_device()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def save_config(args, save_dir, rank=0):
    """保存训练配置"""
    if rank == 0:
        config = {
            'model': 'ResNet-56',
            'dataset': 'CIFAR-100',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'data_augmentation': args.data_aug,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_file = os.path.join(save_dir, 'training_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('分布式环境变量未设置，使用单GPU训练')
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return True, rank, world_size, gpu

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet56(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet56, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(BasicBlock, 16, 9, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 9, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 9, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_data_loaders(batch_size=128, data_augmentation=True, distributed=False, rank=0, world_size=1):
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=100, sampler=test_sampler, 
            num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, train_sampler if distributed else None

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger=None, rank=0, print_freq=100):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % print_freq == 0 and rank == 0:
            current_acc = 100. * correct / total
            msg = f'Epoch {epoch} 批次 [{batch_idx}/{len(train_loader)}] 损失: {loss.item():.4f} 准确率: {current_acc:.2f}%'
            print(msg)
            if logger:
                logger.info(msg)
    
    epoch_time = time.time() - start_time
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    if rank == 0 and logger:
        logger.info(f"Epoch {epoch} 训练完成 - 时间: {epoch_time:.2f}s, 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def test(model, test_loader, criterion, device, epoch, logger=None, distributed=False, rank=0):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    if distributed:
        test_loss_tensor = torch.tensor(test_loss).to(device)
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        test_loss = test_loss_tensor.item()
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    test_time = time.time() - start_time
    test_loss /= len(test_loader) * (dist.get_world_size() if distributed else 1)
    test_acc = 100. * correct / total
    
    if rank == 0 and logger:
        logger.info(f"Epoch {epoch} 测试完成 - 时间: {test_time:.2f}s, 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
    
    return test_loss, test_acc

def save_metrics_to_csv(metrics, save_dir, rank=0):
    """保存训练指标到CSV文件"""
    if rank == 0:
        csv_file = os.path.join(save_dir, 'training_metrics.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'learning_rate'])
            for i, metric in enumerate(metrics):
                writer.writerow([i+1] + metric)

def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path, logger=None, rank=0):
    if rank == 0:
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(state, checkpoint_path)
        if logger:
            logger.info(f"检查点已保存: {checkpoint_path}")

def plot_training_curves(train_losses, train_accs, test_losses, test_accs, save_path='training_curves.png', rank=0):
    if rank == 0:
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs, train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, test_losses, 'r-', label='测试损失')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
        ax2.plot(epochs, test_accs, 'r-', label='测试准确率')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='ResNet-56 CIFAR-100 分布式训练')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='每GPU批大小')
    parser.add_argument('--lr', type=float, default=0.1, help='基础学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine'], help='学习率调度器')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存目录')
    parser.add_argument('--data_aug', action='store_true', default=True, help='启用数据增强')
    parser.add_argument('--print_freq', type=int, default=50, help='打印频率')
    
    args = parser.parse_args()
    
    distributed, rank, world_size, gpu = setup_distributed()
    
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
    logger = setup_logging(args.save_dir, rank)
    
    if rank == 0:
        log_system_info(logger, rank)
        save_config(args, args.save_dir, rank)
        
        if logger:
            logger.info(f'分布式训练: {distributed}')
            logger.info(f'世界大小: {world_size}')
            logger.info(f'使用设备: {device}')
    
    model = ResNet56(num_classes=100).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        msg = f'模型参数量: {total_params:,}'
        print(msg)
        if logger:
            logger.info(msg)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    train_loader, test_loader, train_sampler = get_data_loaders(
        args.batch_size, args.data_aug, distributed, rank, world_size
    )
    
    scaled_lr = args.lr * world_size if distributed else args.lr
    if rank == 0 and logger:
        logger.info(f'原始学习率: {args.lr}, 缩放后学习率: {scaled_lr}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 0
    best_acc = 0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    metrics = []
    
    if args.resume and os.path.isfile(args.resume):
        if rank == 0 and logger:
            logger.info(f"加载检查点 '{args.resume}'")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if distributed else None
        checkpoint = torch.load(args.resume, map_location=map_location)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if rank == 0 and logger:
            logger.info(f"加载检查点完成 (epoch {checkpoint['epoch']})")
    elif args.resume:
        if rank == 0 and logger:
            logger.info(f"找不到检查点 '{args.resume}'")
    
    if rank == 0 and logger:
        logger.info('开始训练...')
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0 and logger:
            logger.info(f'轮次 [{epoch+1}/{args.epochs}] 学习率: {current_lr:.6f}')
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1, logger, rank, args.print_freq)
        test_loss, test_acc = test(model, test_loader, criterion, device, epoch+1, logger, distributed, rank)
        
        scheduler.step()
        
        if rank == 0:
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            metrics.append([train_loss, train_acc, test_loss, test_acc, current_lr])
            
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                best_checkpoint_path = os.path.join(args.save_dir, 'resnet56_best.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_acc, best_checkpoint_path, logger, rank)
                if logger:
                    logger.info(f'新的最佳准确率: {best_acc:.2f}%')
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'resnet56_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_acc, checkpoint_path, logger, rank)
                
                save_metrics_to_csv(metrics, args.save_dir, rank)
    
    if rank == 0:
        training_time = time.time() - start_time
        msg = f'训练完成! 用时: {training_time/3600:.2f} 小时，最佳测试准确率: {best_acc:.2f}%'
        print(msg)
        if logger:
            logger.info(msg)
        
        curves_path = os.path.join(args.save_dir, 'training_curves.png')
        plot_training_curves(train_losses, train_accs, test_losses, test_accs, curves_path, rank)
        if logger:
            logger.info(f'训练曲线已保存至: {curves_path}')
        
        final_checkpoint_path = os.path.join(args.save_dir, 'resnet56_final.pth')
        save_checkpoint(model, optimizer, args.epochs, best_acc, final_checkpoint_path, logger, rank)
        
        save_metrics_to_csv(metrics, args.save_dir, rank)
        if logger:
            logger.info(f'训练指标已保存至: {os.path.join(args.save_dir, "training_metrics.csv")}')
    
    cleanup_distributed()

if __name__ == '__main__':
    main() 