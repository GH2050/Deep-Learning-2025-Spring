import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_patches),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): # x shape: (B, num_patches, dim)
        # Token Mixing
        y = self.norm1(x)       # Apply norm on (B, num_patches, dim) -> last dim is dim
        y = y.transpose(1, 2)   # Shape: (B, dim, num_patches)
        y = self.token_mlp(y)   # MLP acts on num_patches dimension
        y = y.transpose(1, 2)   # Shape: (B, num_patches, dim)
        x = x + y

        # Channel Mixing
        y = self.norm2(x)       # Apply norm on (B, num_patches, dim) -> last dim is dim
        y = self.channel_mlp(y) # MLP acts on dim dimension
        x = x + y
        return x

# 假设 MLPMixerCustom 类已经定义好了
class MLPMixerCustom(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, token_mlp_dim, channel_mlp_dim, dropout=0.):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, dim]
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.norm(x)
        x = x.mean(dim=1) # Global average pooling over patches
        x = self.head(x)
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