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

from src.model import get_model, count_parameters
from src.dataset import get_dataloaders
from src.utils import plot_training_curves, save_experiment_results

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ghostnet_100_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_one_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, total_epochs, logger):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                        disable=not accelerator.is_local_main_process, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        batch_size = inputs.size(0)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        accelerator.backward(loss)
        optimizer.step()
        
        batch_loss = accelerator.gather_for_metrics(loss.detach() * batch_size).sum()
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = accelerator.gather_for_metrics((predicted == targets).sum()).sum()
        batch_total = accelerator.gather_for_metrics(torch.tensor(batch_size, device=accelerator.device)).sum()
        
        total_loss += batch_loss.item()
        correct += batch_correct.item()
        total_samples += batch_total.item()
        
        if accelerator.is_local_main_process:
            current_loss = loss.item()
            current_acc = (predicted == targets).sum().item() / batch_size * 100
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / total_samples
    avg_acc = (correct / total_samples) * 100
    
    if accelerator.is_local_main_process:
        logger.info(f"Epoch {epoch+1} Train - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, accelerator, epoch, total_epochs, logger):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Test]", 
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
        logger.info(f"Epoch {epoch+1} Test - Loss: {avg_loss:.4f}, Top1: {avg_acc_top1:.2f}%, Top5: {avg_acc_top5:.2f}%")
    
    return avg_loss, avg_acc_top1, avg_acc_top5

def main():
    accelerator = Accelerator()
    log_dir = "logs/ghostnet_100"
    logger = setup_logging(log_dir)
    
    if accelerator.is_local_main_process:
        logger.info("="*80)
        logger.info("开始训练 GhostNet-100 模型")
        logger.info("="*80)
        logger.info(f"设备: {accelerator.device}")
        logger.info(f"GPU数量: {accelerator.num_processes}")
    
    config = {
        'model_name': 'ghostnet_100',
        'epochs': 200,
        'batch_size_per_gpu': 64,
        'learning_rate': 0.1,
        'weight_decay': 4e-5,
        'optimizer': 'SGD',
        'scheduler': 'CosineAnnealingLR',
        'momentum': 0.9,
        'use_imagenet_norm': True,
        'num_classes': 100
    }
    
    if accelerator.is_local_main_process:
        logger.info("训练配置:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size_per_gpu'],
        use_imagenet_norm=config['use_imagenet_norm'],
        num_workers=4
    )
    
    if accelerator.is_local_main_process:
        global_batch_size = config['batch_size_per_gpu'] * accelerator.num_processes
        logger.info(f"数据加载器创建完成")
        logger.info(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")
        logger.info(f"每GPU批大小: {config['batch_size_per_gpu']}, 全局批大小: {global_batch_size}")
    
    model = get_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained_timm=True
    )
    
    param_count = count_parameters(model)
    if accelerator.is_local_main_process:
        logger.info(f"模型创建完成: {config['model_name']}")
        logger.info(f"参数量: {param_count:.2f}M")
        logger.info(f"使用预训练权重: True")
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'], 
        momentum=config['momentum'], 
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    if accelerator.is_local_main_process:
        logger.info("Accelerator准备完成")
    
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
    
    if accelerator.is_local_main_process:
        logger.info(f"开始训练，共{config['epochs']}轮")
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process:
            logger.info(f"\nEpoch {epoch+1}/{config['epochs']}, 学习率: {current_lr:.6f}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, accelerator, epoch, config['epochs'], logger
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
                logger.info(f"新的最佳准确率: {best_test_acc:.2f}% (Epoch {best_epoch})")
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_test_acc,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(log_dir, 'best_model.pth'))
            
            if (epoch + 1) % 20 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_test_acc,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        accelerator.wait_for_everyone()
    
    total_time = time.time() - start_time
    
    if accelerator.is_local_main_process:
        logger.info("\n" + "="*80)
        logger.info("训练完成!")
        logger.info(f"总训练时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        logger.info(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch})")
        logger.info(f"最终测试准确率: {test_acc_top1:.2f}%")
        logger.info(f"最终Top-5准确率: {test_acc_top5:.2f}%")
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
            'config': config
        }
        
        save_experiment_results(
            results_data=results_data,
            model_name=config['model_name'],
            hparams=config,
            output_dir=log_dir,
            metrics_history=metrics_history,
            run_label=f"ghostnet_100_final"
        )
        
        plot_training_curves(
            metrics_history, 
            config['model_name'], 
            save_dir=log_dir,
            base_filename='ghostnet_100_training'
        )
        
        logger.info(f"结果已保存到: {log_dir}")

if __name__ == "__main__":
    main() 