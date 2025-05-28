import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class CIFAR100Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'label': label}

def get_cifar100_datasets():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
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
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=None
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=None
    )
    
    trainset = CIFAR100Dataset(train_dataset, transform=transform_train)
    testset = CIFAR100Dataset(test_dataset, transform=transform_test)
    
    return trainset, testset

def get_dataloaders(batch_size=128, num_workers=2):
    trainset, testset = get_cifar100_datasets()
    
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader 