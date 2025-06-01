# train.py
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.cifar100 import get_cifar100_loaders
from models.resnet import ResNet, BasicBlock
from utils.utils import accuracy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torchsummary import summary

def train(epoch, model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += accuracy(outputs, targets)
        total_samples += targets.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}  Acc={avg_acc:.2f}%")

def test(model, testloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_correct += accuracy(outputs, targets)
            total_samples += targets.size(0)
    avg_acc = 100.0 * total_correct / total_samples
    return avg_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-20 or ECA-Net on CIFAR-100')
    parser.add_argument('--model', choices=['resnet20', 'eca_net'], default='resnet20',
                        help='choose model: resnet20 or eca_net')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'cpu'
    trainloader, testloader = get_cifar100_loaders(batch_size=args.batch_size)

    # 构建模型：ResNet-20 或 ECA-Net
    if args.model == 'resnet20':
        model = ResNet(BasicBlock, [3,3,3], use_eca=False)
    else:
        model = ResNet(BasicBlock, [3,3,3], use_eca=True)
    print(f"Model: {summary(model, (3, 32, 32), device=device)}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, 
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    print(f"Start training {args.model} ...")
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        train(epoch, model, trainloader, criterion, optimizer, device)
        scheduler.step()
    elapsed = time.time() - start_time

    acc = test(model, testloader, device)
    print(f"{args.model} Final Test Accuracy: {acc:.2f}%")
    print(f"{args.model} Training Time: {elapsed/60:.2f} minutes")
