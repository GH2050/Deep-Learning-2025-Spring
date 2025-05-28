import torch
import torch.nn as nn
import time
import os
from accelerate import Accelerator

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
    total = 0
    
    for batch_idx, batch in enumerate(trainloader):
        data, target = batch['image'], batch['label']
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        accelerator.backward(loss)
        optimizer.step()
        
        running_loss += loss.item()
        top1, _ = top_k_accuracy(output, target, topk=(1, 5))
        correct_1 += top1
        total += target.size(0)
        
        if batch_idx % 50 == 0:
            current_acc = 100.*correct_1/total
            log_msg = f'Epoch {epoch+1} [{batch_idx:3d}/{len(trainloader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%'
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    return running_loss / len(trainloader), 100.*correct_1 / total

def test_model(model, testloader, criterion, accelerator, log_file):
    model.eval()
    test_loss = 0
    correct_1 = 0
    total = 0
    
    with torch.no_grad():
        for batch in testloader:
            data, target = batch['image'], batch['label']
            output = model(data)
            test_loss += criterion(output, target).item()
            
            top1, _ = top_k_accuracy(output, target, topk=(1, 5))
            correct_1 += top1
            total += target.size(0)
    
    test_loss /= len(testloader)
    acc1 = 100.*correct_1 / total
    
    log_msg = f'测试: Loss: {test_loss:.4f}, Acc: {acc1:.2f}% ({correct_1:.0f}/{total})'
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')
    
    return test_loss, acc1

def train_fast(model, trainloader, testloader, epochs=20, lr=0.1, model_name="model"):
    accelerator = Accelerator()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3, 2*epochs//3], gamma=0.1)
    
    model, optimizer, trainloader, testloader = accelerator.prepare(
        model, optimizer, trainloader, testloader
    )
    
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{model_name}_fast.log'
    
    with open(log_file, 'w') as f:
        f.write(f'{model_name} 快速训练开始 (epochs={epochs})\n')
    
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
        
        log_msg = f'Epoch [{epoch+1:2d}/{epochs}] {epoch_time:.1f}s Train: {train_loss:.4f}/{train_acc:.2f}% Test: {test_loss:.4f}/{test_acc:.2f}% LR: {lr_current:.5f}'
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'logs/{model_name}_best.pth')
    
    final_msg = f'最佳准确率: {best_acc:.2f}%'
    print(final_msg)
    with open(log_file, 'a') as f:
        f.write(final_msg + '\n')
    
    return history, best_acc 