import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision
import torchvision.transforms as transforms
from torchvision.ops import StochasticDepth
from accelerate import Accelerator

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from model import get_model, get_model_info

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MixUpCutMix:
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, batch):
        if random.random() > self.prob:
            return batch
        
        if random.random() < self.switch_prob:
            return self.mixup(batch)
        else:
            return self.cutmix(batch)

    def mixup(self, batch):
        x, y = batch
        batch_size = x.size(0)
        
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)

    def cutmix(self, batch):
        x, y = batch
        batch_size = x.size(0)
        
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size)
        y_a, y_b = y, y[index]
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, (y_a, y_b, lam)

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, target_tuple):
    if isinstance(target_tuple, tuple) and len(target_tuple) == 3:
        y_a, y_b, lam = target_tuple
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    else:
        return criterion(pred, target_tuple)

def get_cifar100_loaders(batch_size=256, num_workers=4, mixup_cutmix=None):
    # CIFAR-100标准化参数
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # 使用AutoAugment
        transforms.ToTensor(),
        normalize
    ])
    
    # 测试数据预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_prob = torch.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def train_epoch(model, train_loader, criterion, optimizer, accelerator, mixup_cutmix=None, 
                stochastic_depth_rate=0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if mixup_cutmix is not None:
            data, target = mixup_cutmix((data, target))
        
        if stochastic_depth_rate > 0:
            for module in model.modules():
                if hasattr(module, 'drop_path_rate'):
                    module.drop_path_rate = stochastic_depth_rate
        
        optimizer.zero_grad()
        output = model(data)
        
        if isinstance(target, tuple):
            loss = mixup_criterion(criterion, output, target)
            _, predicted = output.max(1)
            total += target[0].size(0)
            correct += predicted.eq(target[0]).sum().item()
        else:
            loss = criterion(output, target)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        if accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
            accelerator.print(f'批次 {batch_idx+1}/{len(train_loader)}, 损失: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test_epoch(model, test_loader, criterion, accelerator):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, pred = output.topk(5, 1, True, True)
            total += target.size(0)
            
            target_resized = target.view(-1, 1).expand_as(pred)
            correct = pred.eq(target_resized).float()

            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct[:, :5].sum().item()

    avg_loss = total_loss / len(test_loader)
    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total
    return avg_loss, top1_accuracy, top5_accuracy

def main():
    parser = argparse.ArgumentParser(description='CoAtNet-CIFAROpt训练脚本')
    parser.add_argument('--model_name', type=str, default='coatnet_cifar_opt',
                       choices=['coatnet_cifar_opt', 'coatnet_cifar_opt_large_stem'],
                       help='模型名称 (默认: coatnet_cifar_opt)')
    parser.add_argument('--num_epochs', type=int, default=300, help='训练轮次 (默认: 300)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小 (默认: 128)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率 (默认: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='权重衰减 (默认: 0.02)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='学习率预热轮次 (默认: 10)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数 (默认: 0.1)')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='MixUp alpha值 (默认: 1.0)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha值 (默认: 1.0)')
    parser.add_argument('--stochastic_depth_rate', type=float, default=0.1, help='随机深度失活率 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数 (默认: 4)')
    parser.add_argument('--save_interval', type=int, default=50, help='检查点保存间隔 (默认: 50)')
    parser.add_argument('--log_dir_prefix', type=str, default='logs/coatnet_cifar_opt', help='日志目录名称前缀')
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(args.seed)

    accelerator.print(f"使用设备: {accelerator.device}")

    log_dir_name = f"{args.log_dir_prefix}_{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(os.getcwd(), log_dir_name)
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        accelerator.print(f"日志将保存在: {log_dir}")

    mixup_cutmix_transform = MixUpCutMix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=1.0
    )
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mixup_cutmix=mixup_cutmix_transform if args.mixup_alpha > 0 or args.cutmix_alpha > 0 else None
    )

    model = get_model(args.model_name, num_classes=100)

    if accelerator.is_main_process:
        model_info = get_model_info(accelerator.unwrap_model(model), args.model_name)
        accelerator.print(f"模型信息: {model_info}")

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)
                       help='模型名称')
    parser.add_argument('--num_epochs', type=int, default=300, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热轮次')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Mixup alpha参数')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha参数')
    parser.add_argument('--stochastic_depth_rate', type=float, default=0.1, help='随机深度率')
    parser.add_argument('--save_interval', type=int, default=50, help='保存间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 参数设置
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    warmup_epochs = args.warmup_epochs
    label_smoothing = args.label_smoothing
    mixup_alpha = args.mixup_alpha
    cutmix_alpha = args.cutmix_alpha
    stochastic_depth_rate = args.stochastic_depth_rate
    save_interval = args.save_interval
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = get_model(model_name, num_classes=100)
    model = model.to(device)
    
    # 打印模型信息
    model_info = get_model_info(model, model_name)
    print(f"模型信息: {model_info}")
    
    # 数据加载器
    mixup_cutmix = MixUpCutMix(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, 
                               prob=0.8, switch_prob=0.5)
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, 
                                                    mixup_cutmix=None)  # 后续在训练中应用
    
    # 损失函数和优化器
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器（预热 + 余弦退火）
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup_scheduler, cosine_scheduler], 
                           milestones=[warmup_epochs])
    
    # 训练历史记录
    train_losses, train_accs = [], []
    test_losses, test_top1_accs, test_top5_accs = [], [], []
    best_acc = 0
    best_epoch = 0
    
    # 创建日志目录
    log_dir = f"logs/{model_name}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"开始训练 {model_name}")
    print(f"总轮次: {num_epochs}, 批次大小: {batch_size}")
    print(f"学习率: {learning_rate}, 权重衰减: {weight_decay}")
    print(f"标签平滑: {label_smoothing}, 随机深度: {stochastic_depth_rate}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                          device, mixup_cutmix, stochastic_depth_rate)
        
        # 测试
        test_loss, test_top1_acc, test_top5_acc = test_epoch(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_top1_accs.append(test_top1_acc)
        test_top5_accs.append(test_top5_acc)
        
        # 保存最佳模型
        if test_top1_acc > best_acc:
            best_acc = test_top1_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_info': model_info
            }, f'{log_dir}/best_model.pth')
        
        # 打印进度
        epoch_time = time.time() - start_time
        print(f'轮次 {epoch+1:3d}/{num_epochs} | '
              f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
              f'测试损失: {test_loss:.4f} | Top-1: {test_top1_acc:.2f}% | Top-5: {test_top5_acc:.2f}% | '
              f'学习率: {current_lr:.6f} | 时间: {epoch_time:.1f}s')
        
        # 每save_interval轮保存一次检查点
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_top1_accs': test_top1_accs,
                'test_top5_accs': test_top5_accs,
                'model_info': model_info
            }, f'{log_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    print(f"\n训练完成！")
    print(f"最佳准确率: {best_acc:.2f}% (轮次 {best_epoch+1})")
    print(f"日志保存在: {log_dir}")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_top1_accs, label='测试Top-1准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.subplot(1, 3, 3)
    plt.plot(test_top5_accs, label='测试Top-5准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('Top-5准确率曲线')
    
    plt.tight_layout()
    plt.savefig(f'{log_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存训练历史
    np.savez(f'{log_dir}/training_history.npz',
             train_losses=train_losses,
             train_accs=train_accs,
             test_losses=test_losses,
             test_top1_accs=test_top1_accs,
             test_top5_accs=test_top5_accs)

if __name__ == '__main__':
    main() 