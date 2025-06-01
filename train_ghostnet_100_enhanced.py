import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import tqdm
import time
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import random

from src.model import get_model, count_parameters
from src.dataset import get_dataloaders
from src.utils import plot_training_curves, save_experiment_results

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ghostnet_100_enhanced_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整lambda以匹配实际的混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_one_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, total_epochs, logger, 
                   mixup_alpha=1.0, cutmix_alpha=1.0, mixup_prob=0.5):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [训练]", 
                        disable=not accelerator.is_local_main_process, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        batch_size = inputs.size(0)
        
        # 随机选择数据增强策略
        use_mixup = random.random() < mixup_prob
        use_cutmix = random.random() < 0.5 and not use_mixup
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup or use_cutmix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        accelerator.backward(loss)
        optimizer.step()
        
        batch_loss = accelerator.gather_for_metrics(loss.detach() * batch_size).sum()
        
        # 计算准确率（对于mixup/cutmix使用原始标签）
        if use_mixup or use_cutmix:
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = accelerator.gather_for_metrics((predicted == targets_a).sum() * lam + (predicted == targets_b).sum() * (1 - lam)).sum()
        else:
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = accelerator.gather_for_metrics((predicted == targets).sum()).sum()
        
        batch_total = accelerator.gather_for_metrics(torch.tensor(batch_size, device=accelerator.device)).sum()
        
        total_loss += batch_loss.item()
        correct += batch_correct.item()
        total_samples += batch_total.item()
        
        if accelerator.is_local_main_process:
            current_loss = loss.item()
            if use_mixup or use_cutmix:
                _, predicted = torch.max(outputs.data, 1)
                current_acc = ((predicted == targets_a).sum() * lam + (predicted == targets_b).sum() * (1 - lam)).item() / batch_size * 100
            else:
                current_acc = (predicted == targets).sum().item() / batch_size * 100
            
            aug_type = "Mixup" if use_mixup else ("CutMix" if use_cutmix else "Normal")
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%", aug=aug_type)
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%, Aug: {aug_type}")
    
    avg_loss = total_loss / total_samples
    avg_acc = (correct / total_samples) * 100
    
    if accelerator.is_local_main_process:
        logger.info(f"Epoch {epoch+1} 训练 - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, accelerator, epoch, total_epochs, logger):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{total_epochs} [测试]", 
                        disable=not accelerator.is_local_main_process, leave=False)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            batch_size = inputs.size(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_loss = accelerator.gather_for_metrics(loss.detach() * batch_size).sum()
            all_outputs = accelerator.gather_for_metrics(outputs)
            all_targets = accelerator.gather_for_metrics(targets)
            
            total_loss += batch_loss.item()
            total_samples += all_targets.size(0)
            
            _, predicted_top1 = torch.max(all_outputs.data, 1)
            correct_top1 += (predicted_top1 == all_targets).sum().item()
            
            _, predicted_top5 = torch.topk(all_outputs.data, 5, dim=1)
            target_expanded = all_targets.view(-1, 1).expand_as(predicted_top5)
            correct_top5 += torch.sum(predicted_top5 == target_expanded).item()
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / total_samples
    avg_acc_top1 = (correct_top1 / total_samples) * 100
    avg_acc_top5 = (correct_top5 / total_samples) * 100
    
    if accelerator.is_local_main_process:
        logger.info(f"Epoch {epoch+1} 测试 - Loss: {avg_loss:.4f}, Top1: {avg_acc_top1:.2f}%, Top5: {avg_acc_top5:.2f}%")
    
    return avg_loss, avg_acc_top1, avg_acc_top5

def main():
    accelerator = Accelerator()
    log_dir = "logs/ghostnet_100_enhanced"
    logger = setup_logging(log_dir)
    
    if accelerator.is_local_main_process:
        logger.info("="*80)
        logger.info("GhostNet-100 增强训练 (Mixup + CutMix + AutoAugment)")
        logger.info("="*80)
        logger.info(f"设备: {accelerator.device}")
        logger.info(f"GPU数量: {accelerator.num_processes}")
    
    config = {
        'model_name': 'ghostnet_100',
        'epochs': 300,  # 增加到300轮
        'batch_size_per_gpu': 128,
        'learning_rate': 0.1,
        'weight_decay': 4e-5,
        'optimizer': 'SGD',
        'scheduler': 'CosineAnnealingLR',
        'momentum': 0.9,
        'use_imagenet_norm': True,
        'num_classes': 100,
        'eta_min': 1e-6,
        'target_accuracy': 75.71,
        'expected_params_M': 4.03,
        # 数据增强参数
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,
        'warmup_epochs': 10,
    }
    
    if accelerator.is_local_main_process:
        logger.info("训练配置 (增强版):")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size_per_gpu'],
        use_imagenet_norm=config['use_imagenet_norm'],
        num_workers=min(8, os.cpu_count())
    )
    
    if accelerator.is_local_main_process:
        global_batch_size = config['batch_size_per_gpu'] * accelerator.num_processes
        logger.info(f"数据加载器创建完成")
        logger.info(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")
        logger.info(f"每GPU批大小: {config['batch_size_per_gpu']}, 全局批大小: {global_batch_size}")
    
    model = get_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained_timm=False  # 暂时关闭预训练权重
    )
    
    param_count = count_parameters(model)
    if accelerator.is_local_main_process:
        logger.info(f"模型创建完成: {config['model_name']} (使用预训练权重)")
        logger.info(f"实际参数量: {param_count:.2f}M (预期: {config['expected_params_M']}M)")
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'], 
        momentum=config['momentum'], 
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'], 
        eta_min=config['eta_min']
    )
    
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    metrics_history = {
        'train_losses': [], 
        'train_accs': [], 
        'test_losses': [], 
        'test_accs': [], 
        'test_accs_top5': [],
        'learning_rates': []
    }
    
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process:
            logger.info(f"\nEpoch {epoch+1}/{config['epochs']}, 学习率: {current_lr:.6f}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, accelerator, epoch, config['epochs'], logger,
            mixup_alpha=config['mixup_alpha'],
            cutmix_alpha=config['cutmix_alpha'],
            mixup_prob=config['mixup_prob']
        )
        
        test_loss, test_acc_top1, test_acc_top5 = evaluate(
            model, test_loader, criterion, accelerator, epoch, config['epochs'], logger
        )
        
        scheduler.step()
        
        if accelerator.is_local_main_process:
            metrics_history['train_losses'].append(train_loss)
            metrics_history['train_accs'].append(train_acc)
            metrics_history['test_losses'].append(test_loss)
            metrics_history['test_accs'].append(test_acc_top1)
            metrics_history['test_accs_top5'].append(test_acc_top5)
            metrics_history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1} 完成，耗时: {epoch_time:.2f}秒")
            
            if test_acc_top1 > best_test_acc:
                best_test_acc = test_acc_top1
                best_epoch = epoch + 1
                logger.info(f"🎉 新的最佳准确率: {best_test_acc:.2f}% (Epoch {best_epoch})")
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_test_acc,
                    'config': config
                }, os.path.join(log_dir, 'best_model.pth'))
                
                if best_test_acc >= config['target_accuracy']:
                    logger.info(f"🎉 达到目标准确率 {config['target_accuracy']}%! 当前最佳: {best_test_acc:.2f}%")
            
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_test_acc,
                    'config': config
                }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                logger.info(f"💾 保存检查点: checkpoint_epoch_{epoch+1}.pth")
        
        accelerator.wait_for_everyone()
    
    total_time = time.time() - start_time
    
    if accelerator.is_local_main_process:
        logger.info("\n" + "="*80)
        logger.info("训练完成!")
        logger.info(f"总训练时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        logger.info(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch})")
        logger.info(f"最终测试准确率: {test_acc_top1:.2f}%")
        logger.info(f"最终Top-5准确率: {test_acc_top5:.2f}%")
        logger.info(f"目标准确率: {config['target_accuracy']}%")
        
        if best_test_acc >= config['target_accuracy']:
            logger.info("✅ 成功达到目标准确率!")
        else:
            gap = config['target_accuracy'] - best_test_acc
            logger.info(f"❌ 距离目标准确率还差: {gap:.2f}%")
        
        logger.info("="*80)
        
        results_data = {
            'model_name': config['model_name'],
            'best_test_acc_top1': best_test_acc,
            'final_test_acc_top1': test_acc_top1,
            'final_test_acc_top5': test_acc_top5,
            'best_epoch': best_epoch,
            'total_epochs': config['epochs'],
            'total_training_time_hours': total_time / 3600,
            'parameters_M': param_count,
            'target_accuracy': config['target_accuracy'],
            'achieved_target': best_test_acc >= config['target_accuracy'],
            'config': config
        }
        
        save_experiment_results(
            results_data=results_data,
            model_name=config['model_name'] + "_enhanced",
            hparams=config,
            output_dir=log_dir,
            metrics_history=metrics_history,
            run_label=f"ghostnet_100_enhanced"
        )
        
        plot_training_curves(
            metrics_history, 
            config['model_name'] + "_enhanced", 
            save_dir=log_dir,
            base_filename='ghostnet_100_enhanced_training'
        )
        
        logger.info(f"结果已保存到: {log_dir}")

if __name__ == "__main__":
    main() 