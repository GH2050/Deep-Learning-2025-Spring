import torch
import torch.nn as nn
import time
import os
import datetime
import json
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse

def get_gpu_memory_info():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name()}, 内存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB"
    return "CPU Only"

def log_system_info(accelerator, log_file):
    """记录系统和训练配置信息"""
    with open(log_file, 'a') as f:
        f.write(f'='*80 + '\n')
        f.write(f'训练开始时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'='*80 + '\n')
        f.write(f'硬件信息:\n')
        f.write(f'  - 使用设备: {accelerator.device}\n')
        f.write(f'  - 进程数: {accelerator.num_processes}\n')
        f.write(f'  - 混合精度: {accelerator.mixed_precision}\n')
        f.write(f'  - {get_gpu_memory_info()}\n')
        f.write(f'  - PyTorch版本: {torch.__version__}\n')
        f.write(f'-'*50 + '\n')

def log_training_config(config_dict, log_file):
    """记录训练配置"""
    with open(log_file, 'a') as f:
        f.write(f'训练配置:\n')
        for key, value in config_dict.items():
            f.write(f'  - {key}: {value}\n')
        f.write(f'-'*50 + '\n')

def top_k_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

def train_epoch(model, trainloader, optimizer, criterion, accelerator, epoch, log_file=None):
    model.train()
    running_loss = 0.0
    correct_1 = 0
    total = 0
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(trainloader):
        batch_start_time = time.time()
        data, target = batch['image'], batch['label']
        
        optimizer.zero_grad()
        
        with accelerator.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        accelerator.backward(loss)
        optimizer.step()
        
        batch_time = time.time() - batch_start_time
        running_loss += loss.item()
        top1, top5 = top_k_accuracy(output, target, topk=(1, 5))
        correct_1 += top1
        total += target.size(0)
        
        if batch_idx % 30 == 0 and accelerator.is_main_process:
            current_acc = 100.*correct_1/total
            lr = optimizer.param_groups[0]['lr']
            log_msg = f'[{datetime.datetime.now().strftime("%H:%M:%S")}] Epoch {epoch+1:2d} [{batch_idx:3d}/{len(trainloader)}] Loss: {loss.item():.4f} Top1: {current_acc:.2f}% Top5: {100.*top5/target.size(0):.2f}% LR: {lr:.6f} Time: {batch_time:.3f}s'
            print(log_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(log_msg + '\n')
    
    epoch_time = time.time() - epoch_start_time
    
    if accelerator.num_processes > 1:
        running_loss_tensor = torch.tensor(running_loss / len(trainloader), device=accelerator.device)
        correct_1_tensor = torch.tensor(correct_1, device=accelerator.device)
        total_tensor = torch.tensor(total, device=accelerator.device)
        
        running_loss = accelerator.gather_for_metrics(running_loss_tensor).mean().item()
        correct_1 = accelerator.gather_for_metrics(correct_1_tensor).sum().item()
        total = accelerator.gather_for_metrics(total_tensor).sum().item()
    else:
        running_loss = running_loss / len(trainloader)
    
    return running_loss, 100.*correct_1 / total, epoch_time

def test_model(model, testloader, criterion, accelerator, log_file=None):
    model.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    test_start_time = time.time()
    
    with torch.no_grad():
        for batch in testloader:
            data, target = batch['image'], batch['label']
            
            with accelerator.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            test_loss += loss.item()
            top1, top5 = top_k_accuracy(output, target, topk=(1, 5))
            correct_1 += top1
            correct_5 += top5
            total += target.size(0)
    
    test_time = time.time() - test_start_time
    
    if accelerator.num_processes > 1:
        test_loss_tensor = torch.tensor(test_loss / len(testloader), device=accelerator.device)
        correct_1_tensor = torch.tensor(correct_1, device=accelerator.device)
        correct_5_tensor = torch.tensor(correct_5, device=accelerator.device)
        total_tensor = torch.tensor(total, device=accelerator.device)
        
        test_loss = accelerator.gather_for_metrics(test_loss_tensor).mean().item()
        correct_1 = accelerator.gather_for_metrics(correct_1_tensor).sum().item()
        correct_5 = accelerator.gather_for_metrics(correct_5_tensor).sum().item()
        total = accelerator.gather_for_metrics(total_tensor).sum().item()
    else:
        test_loss = test_loss / len(testloader)
    
    acc1 = 100.*correct_1 / total
    acc5 = 100.*correct_5 / total
    
    if accelerator.is_main_process:
        log_msg = f'[{datetime.datetime.now().strftime("%H:%M:%S")}] 测试结果: Loss: {test_loss:.4f}, Top1: {acc1:.2f}%, Top5: {acc5:.2f}% ({correct_1:.0f}/{total}) Time: {test_time:.1f}s'
        print(log_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    return test_loss, acc1, acc5

def train_with_accelerate(model, trainloader, testloader, epochs=20, lr=0.1, model_name="model", **kwargs):
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with="all",
        project_dir="logs/"
    )
    
    set_seed(42)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4, 
        nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )
    
    if accelerator.is_main_process:
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/{model_name}_accelerate_{timestamp}.log'
        
        log_system_info(accelerator, log_file)
        
        config = {
            'model': model_name,
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': kwargs.get('batch_size', 'unknown'),
            'optimizer': 'SGD',
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler': 'CosineAnnealingLR',
            'mixed_precision': 'fp16',
            'seed': 42
        }
        log_training_config(config, log_file)
        
        with open(log_file, 'a') as f:
            f.write(f'模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n')
            f.write(f'训练集大小: {len(trainloader.dataset)}\n')
            f.write(f'测试集大小: {len(testloader.dataset)}\n')
            f.write(f'='*80 + '\n')
    else:
        log_file = None
    
    best_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'test_acc5': [],
        'epoch_times': [], 'learning_rates': []
    }
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc, epoch_time = train_epoch(
            model, trainloader, optimizer, criterion, accelerator, epoch, log_file
        )
        test_loss, test_acc, test_acc5 = test_model(
            model, testloader, criterion, accelerator, log_file
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_acc5'].append(test_acc5)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        lr_current = optimizer.param_groups[0]['lr']
        
        if accelerator.is_main_process:
            log_msg = f'Epoch [{epoch+1:2d}/{epochs}] {epoch_time:.1f}s Train: {train_loss:.4f}/{train_acc:.2f}% Test: {test_loss:.4f}/{test_acc:.2f}%/{test_acc5:.2f}% LR: {lr_current:.6f}'
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                accelerator.save_model(model, f'logs/{model_name}_best_accelerate_{timestamp}')
                with open(log_file, 'a') as f:
                    f.write(f'*** 新的最佳模型保存! 准确率: {best_acc:.2f}% ***\n')
        
        accelerator.wait_for_everyone()
    
    total_training_time = time.time() - training_start_time
    
    if accelerator.is_main_process:
        with open(log_file, 'a') as f:
            f.write(f'='*80 + '\n')
            f.write(f'训练完成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'总训练时间: {total_training_time:.1f}s ({total_training_time/60:.1f}分钟)\n')
            f.write(f'平均每轮时间: {total_training_time/epochs:.1f}s\n')
            f.write(f'最佳准确率: {best_acc:.2f}% (第{best_epoch}轮)\n')
            f.write(f'='*80 + '\n')
        
        history_file = f'logs/{model_name}_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        final_msg = f'训练完成! 最佳准确率: {best_acc:.2f}% (第{best_epoch}轮)'
        print(final_msg)
    
    return history, best_acc

def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Accelerate训练')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--model', type=str, default='resnet_20', 
                       choices=['resnet_20', 'eca_resnet_20', 'ghost_resnet_20', 'convnext_tiny'],
                       help='模型选择')
    args = parser.parse_args()
    
    from dataset import get_dataloaders
    from model import resnet20, eca_resnet20, ghost_resnet20, convnext_tiny
    
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=4)
    
    model_map = {
        'resnet_20': resnet20,
        'eca_resnet_20': eca_resnet20,
        'ghost_resnet_20': ghost_resnet20,
        'convnext_tiny': convnext_tiny
    }
    
    model = model_map[args.model]()
    
    history, best_acc = train_with_accelerate(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        lr=args.lr,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    return history, best_acc

if __name__ == "__main__":
    main() 