import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 假设 MLPMixerCustom 类已经定义好了
class MLPMixerCustom(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=128, depth=4, 
                 token_mlp_dim=128, channel_mlp_dim=256, num_classes=100):
        super(MLPMixerCustom, self).__init__()
        # 这里需要实现具体的模型结构
        pass

    def forward(self, x):
        # 这里需要实现前向传播逻辑
        return x

# 注册模型构建函数
def mlp_mixer_tiny_custom_builder(num_classes=100, **kwargs):
    return MLPMixerCustom(image_size=32, patch_size=4, dim=128, depth=4, 
                          token_mlp_dim=128, channel_mlp_dim=256,
                          num_classes=num_classes, **kwargs)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 构建模型
model = mlp_mixer_tiny_custom_builder(num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')