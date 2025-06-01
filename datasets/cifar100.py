# datasets/cifar100.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loaders(batch_size=128): # 128
    """
    返回 CIFAR-100 的训练集和测试集 DataLoader。
    使用随机裁剪、翻转等增强。
    """
    # CIFAR-100 数据集的均值和标准差
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, 
                                 download=True, transform=transform_train)
    testset  = datasets.CIFAR100(root='./data', train=False, 
                                 download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    testloader  = DataLoader(testset, batch_size=batch_size, 
                             shuffle=False, num_workers=2)
    return trainloader, testloader
