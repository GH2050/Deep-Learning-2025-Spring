#!/usr/bin/env python3

import torch
from accelerate import Accelerator
from src.dataset import get_dataloaders
from src.model import resnet20
import time

def main():
    accelerator = Accelerator(mixed_precision='fp16')
    
    print(f"设备: {accelerator.device}")
    print(f"进程数: {accelerator.num_processes}")
    print(f"混合精度: {accelerator.mixed_precision}")
    
    print("加载数据和模型...")
    trainloader, testloader = get_dataloaders(batch_size=128, num_workers=2)
    model = resnet20()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)
    
    print("开始快速测试训练...")
    model.train()
    start_time = time.time()
    
    for batch_idx, batch in enumerate(trainloader):
        if batch_idx >= 10:
            break
            
        data, target = batch['image'], batch['label']
        
        optimizer.zero_grad()
        
        with accelerator.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        accelerator.backward(loss)
        optimizer.step()
        
        if accelerator.is_main_process:
            print(f"批次 {batch_idx}: 损失 {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"10个批次训练完成，耗时: {elapsed:.2f}秒")
    print("Accelerate 测试成功!")

if __name__ == "__main__":
    main() 