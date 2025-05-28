import torch
import torch.nn as nn
import time
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse

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
    
    for batch_idx, batch in enumerate(trainloader):
        data, target = batch['image'], batch['label']
        
        optimizer.zero_grad()
        
        with accelerator.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        accelerator.backward(loss)
        optimizer.step()
        
        running_loss += loss.item()
        top1, _ = top_k_accuracy(output, target, topk=(1, 5))
        correct_1 += top1
        total += target.size(0)
        
        if batch_idx % 30 == 0 and accelerator.is_main_process:
            current_acc = 100.*correct_1/total
            log_msg = f'进程{accelerator.process_index} Epoch {epoch+1} [{batch_idx:3d}/{len(trainloader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%'
            print(log_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(log_msg + '\n')
    
    running_loss = accelerator.gather_for_metrics(torch.tensor(running_loss / len(trainloader))).mean().item()
    correct_1 = accelerator.gather_for_metrics(torch.tensor(correct_1)).sum().item()
    total = accelerator.gather_for_metrics(torch.tensor(total)).sum().item()
    
    return running_loss, 100.*correct_1 / total

def test_model(model, testloader, criterion, accelerator, log_file=None):
    model.eval()
    test_loss = 0
    correct_1 = 0
    total = 0
    
    with torch.no_grad():
        for batch in testloader:
            data, target = batch['image'], batch['label']
            
            with accelerator.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            test_loss += loss.item()
            top1, _ = top_k_accuracy(output, target, topk=(1, 5))
            correct_1 += top1
            total += target.size(0)
    
    test_loss = accelerator.gather_for_metrics(torch.tensor(test_loss / len(testloader))).mean().item()
    correct_1 = accelerator.gather_for_metrics(torch.tensor(correct_1)).sum().item()
    total = accelerator.gather_for_metrics(torch.tensor(total)).sum().item()
    
    acc1 = 100.*correct_1 / total
    
    if accelerator.is_main_process:
        log_msg = f'测试结果: Loss: {test_loss:.4f}, Acc: {acc1:.2f}% ({correct_1:.0f}/{total})'
        print(log_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    return test_loss, acc1

def train_with_accelerate(model, trainloader, testloader, epochs=20, lr=0.1, model_name="model"):
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
        log_file = f'logs/{model_name}_accelerate.log'
        with open(log_file, 'w') as f:
            f.write(f'{model_name} Accelerate训练开始\n')
            f.write(f'使用设备: {accelerator.device}\n')
            f.write(f'进程数: {accelerator.num_processes}\n')
            f.write(f'混合精度: {accelerator.mixed_precision}\n')
            f.write(f'参数: epochs={epochs}, lr={lr}\n')
            f.write('-' * 50 + '\n')
    else:
        log_file = None
    
    best_acc = 0
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, accelerator, epoch, log_file)
        test_loss, test_acc = test_model(model, testloader, criterion, accelerator, log_file)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        epoch_time = time.time() - start_time
        lr_current = optimizer.param_groups[0]['lr']
        
        if accelerator.is_main_process:
            log_msg = f'Epoch [{epoch+1:2d}/{epochs}] {epoch_time:.1f}s Train: {train_loss:.4f}/{train_acc:.2f}% Test: {test_loss:.4f}/{test_acc:.2f}% LR: {lr_current:.5f}'
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            if test_acc > best_acc:
                best_acc = test_acc
                accelerator.save_model(model, f'logs/{model_name}_best_accelerate')
        
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        final_msg = f'最佳准确率: {best_acc:.2f}%'
        print(final_msg)
        with open(log_file, 'a') as f:
            f.write(final_msg + '\n')
    
    return history, best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='resnet_20', choices=['resnet_20', 'eca_resnet_20'])
    args = parser.parse_args()
    
    from dataset import get_dataloaders
    from model import resnet20, eca_resnet20
    
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=4)
    
    if args.model == 'resnet_20':
        model = resnet20()
    elif args.model == 'eca_resnet_20':
        model = eca_resnet20()
    
    history, best_acc = train_with_accelerate(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        lr=args.lr,
        model_name=args.model
    )
    
    return history, best_acc

if __name__ == "__main__":
    main() 