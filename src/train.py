import torch
import torch.nn as nn
import time
import os
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

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

def train_epoch(model, trainloader, optimizer, criterion, accelerator, epoch, log_file):
    model.train()
    running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    for batch_idx, batch in enumerate(trainloader):
        data, target = batch['image'], batch['label']
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        accelerator.backward(loss)
        optimizer.step()
        
        running_loss += loss.item()
        top1, top5 = top_k_accuracy(output, target, topk=(1, 5))
        correct_1 += top1
        correct_5 += top5
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            log_msg = f'Epoch {epoch+1} 训练批次 [{batch_idx}/{len(trainloader)}] 损失: {loss.item():.4f} Top-1: {100.*correct_1/total:.2f}% Top-5: {100.*correct_5/total:.2f}%'
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    return running_loss / len(trainloader), 100.*correct_1 / total, 100.*correct_5 / total

def test(model, testloader, criterion, accelerator, log_file):
    model.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for batch in testloader:
            data, target = batch['image'], batch['label']
            output = model(data)
            test_loss += criterion(output, target).item()
            
            top1, top5 = top_k_accuracy(output, target, topk=(1, 5))
            correct_1 += top1
            correct_5 += top5
            total += target.size(0)
    
    test_loss /= len(testloader)
    acc1 = 100.*correct_1 / total
    acc5 = 100.*correct_5 / total
    
    log_msg = f'测试结果: 平均损失: {test_loss:.4f}, Top-1 准确率: {acc1:.2f}% ({correct_1:.0f}/{total}), Top-5 准确率: {acc5:.2f}% ({correct_5:.0f}/{total})'
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')
    
    return test_loss, acc1, acc5

def train_model(model, trainloader, testloader, epochs=100, lr=0.1, model_name="resnet", warmup_steps=100):
    accelerator = Accelerator()
    
    criterion = nn.CrossEntropyLoss()
    
    if model_name.startswith('convnext'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    total_steps = epochs * len(trainloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )
    
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{model_name}_training.log'
    
    with open(log_file, 'w') as f:
        f.write(f'{model_name} 训练开始\n')
        f.write(f'参数: epochs={epochs}, lr={lr}, total_steps={total_steps}\n')
        f.write('-' * 50 + '\n')
    
    best_acc = 0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc1, train_acc5 = train_epoch(model, trainloader, optimizer, criterion, accelerator, epoch, log_file)
        test_loss, test_acc1, test_acc5 = test(model, testloader, criterion, accelerator, log_file)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc1)
        test_accs.append(test_acc1)
        
        epoch_time = time.time() - start_time
        
        log_msg = f'轮次 [{epoch+1}/{epochs}] 时间: {epoch_time:.1f}s 训练损失: {train_loss:.4f} 测试损失: {test_loss:.4f} 训练Top-1: {train_acc1:.2f}% 测试Top-1: {test_acc1:.2f}% 学习率: {optimizer.param_groups[0]["lr"]:.6f}'
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        if test_acc1 > best_acc:
            best_acc = test_acc1
            accelerator.save(model.state_dict(), f'logs/{model_name}_best_model.pth')
    
    final_msg = f'最佳测试准确率: {best_acc:.2f}%'
    print(final_msg)
    with open(log_file, 'a') as f:
        f.write(final_msg + '\n')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc
    } 