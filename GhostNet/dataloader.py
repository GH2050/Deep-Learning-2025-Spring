import torch
import torchvision
import torchvision.transforms as transforms


# 设置标准归一化参数（与 CIFAR-100 官方统计一致）
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# 数据预处理：张量化 + 标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

# 批大小
batch_size = 4

# 加载 CIFAR-100 训练集
trainset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

# 加载 CIFAR-100 测试集
testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


classes = trainset.classes  # classes = testset.classes

# 简单测试
if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch: {labels}")
    print(f"First 4 labels (as names): {[classes[l] for l in labels]}")
    print(classes)