#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import argparse
import sys

sys.path.append('src')
from model import get_model
from dataset import get_cifar100_datasets
from utils import (
    get_hyperparameters, plot_training_curves, save_experiment_results,
    setup_logging, log_system_info, setup_distributed, cleanup_distributed,
    get_optimizer_scheduler, train_one_epoch, test_model, save_checkpoint
)

def get_data_loaders(hparams, distributed=False, rank=0, world_size=1):
    """获取数据加载器"""
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    
    train_dataset, test_dataset = get_cifar100_datasets(
        data_dir='./data',
        use_imagenet_norm=hparams['use_imagenet_norm']
    )
    
    batch_size = hparams['batch_size_per_gpu']
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size*2, sampler=test_sampler,
            num_workers=4, pin_memory=True
        )
        return train_loader, test_loader, train_sampler
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size*2, shuffle=False,
            num_workers=4, pin_memory=True
        )
        return train_loader, test_loader, None

def main():
    parser = argparse.ArgumentParser(description='ResNet-56 CIFAR-100 改进训练')
    parser.add_argument('--model_name', type=str, default='resnet_56', help='模型名称')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='./logs/resnet56_improved', help='保存目录')
    parser.add_argument('--print_freq', type=int, default=50, help='打印频率')
    parser.add_argument('--use_mixup', action='store_true', default=True, help='使用Mixup数据增强')
    
    args = parser.parse_args()
    
    distributed, rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
    logger = setup_logging(args.save_dir, rank)
    
    if rank == 0:
        log_system_info(logger, rank)
        
        if logger:
            logger.info(f'分布式训练: {distributed}')
            logger.info(f'世界大小: {world_size}')
            logger.info(f'使用设备: {device}')
    
    hparams = get_hyperparameters(args.model_name)
    hparams['epochs'] = args.epochs
    
    if rank == 0 and logger:
        import json
        logger.info(f"使用超参数配置: {json.dumps(hparams, indent=2, ensure_ascii=False)}")
    
    model = get_model(
        args.model_name, 
        num_classes=100, 
        **hparams.get('model_constructor_params', {})
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        msg = f'模型参数量: {total_params:,}'
        print(msg)
        if logger:
            logger.info(msg)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    train_loader, test_loader, train_sampler = get_data_loaders(hparams, distributed, rank, world_size)
    
    scaled_lr = hparams['lr'] * world_size if distributed else hparams['lr']
    hparams['lr'] = scaled_lr
    
    if rank == 0 and logger:
        logger.info(f'缩放后学习率: {scaled_lr}')
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = get_optimizer_scheduler(model, hparams, len(train_loader))
    
    start_epoch = 0
    best_acc = 0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
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
    
    if rank == 0 and logger:
        logger.info('开始训练...')
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0 and logger:
            logger.info(f'轮次 [{epoch+1}/{args.epochs}] 学习率: {current_lr:.6f}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch+1, logger, rank, args.use_mixup
        )
        test_loss, test_acc = test_model(model, test_loader, criterion, device, epoch+1, logger, distributed, rank)
        
        scheduler.step()
        
        if rank == 0:
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                best_checkpoint_path = os.path.join(args.save_dir, 'resnet56_best.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_acc, best_checkpoint_path, logger, rank)
                if logger:
                    logger.info(f'新的最佳准确率: {best_acc:.2f}%')
            
            if (epoch + 1) % 20 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'resnet56_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_acc, checkpoint_path, logger, rank)
    
    if rank == 0:
        training_time = time.time() - start_time
        msg = f'训练完成! 用时: {training_time/3600:.2f} 小时，最佳测试准确率: {best_acc:.2f}%'
        print(msg)
        if logger:
            logger.info(msg)
        
        metrics_history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs
        }
        
        results_data = {
            'best_accuracy': best_acc,
            'final_accuracy': test_accs[-1] if test_accs else 0,
            'training_time_hours': training_time / 3600,
            'total_epochs': args.epochs,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        save_experiment_results(
            results_data=results_data,
            model_name=args.model_name,
            hparams=hparams,
            output_dir=args.save_dir,
            metrics_history=metrics_history,
            run_label='improved'
        )
        
        curves_path = os.path.join(args.save_dir, 'training_curves.png')
        plot_training_curves(metrics_history, args.model_name, args.save_dir, 'resnet56_improved')
        if logger:
            logger.info(f'训练曲线已保存至: {curves_path}')
        
        final_checkpoint_path = os.path.join(args.save_dir, 'resnet56_final.pth')
        save_checkpoint(model, optimizer, args.epochs, best_acc, final_checkpoint_path, logger, rank)
    
    cleanup_distributed()

if __name__ == '__main__':
    main() 