import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

# 权重归一化实现
class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(out_features))
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        weight = F.normalize(self.weight_v, dim=1) * self.weight_g.unsqueeze(1)
        return F.linear(x, weight, self.bias)

# Swish自适应激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 门控网络，用于MoE选择专家
class GatingNetwork(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = WeightNormLinear(dim, num_experts)
        
    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

# 专家网络
class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation=Swish()):
        super().__init__()
        self.mlp = nn.Sequential(
            WeightNormLinear(in_dim, hidden_dim),
            activation,
            WeightNormLinear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

# 混合专家层
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, hidden_dim=None, activation=Swish(), capacity_factor=1.2):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, dim, activation) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(dim, num_experts)
        self.capacity_factor = capacity_factor
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        
        # 计算门控权重
        gates = self.gating(x_flat)  # [batch_size*seq_len, num_experts]
        
        # 选择top-k专家
        top_k = min(2, self.num_experts)
        top_values, top_indices = torch.topk(gates, top_k, dim=-1)
        
        # 专家路由
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (top_indices == i).any(dim=1).float().unsqueeze(1)
            if mask.sum() > 0:
                expert_input = x_flat * mask
                expert_outputs += mask * self.experts[i](expert_input)
        
        return expert_outputs.reshape(batch_size, seq_len, dim)

# 优化后的MixerBlock
class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout=0., 
                 use_moe=False, num_experts=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # 使用权重归一化和Swish激活的Token MLP
        self.token_mlp = nn.Sequential(
            WeightNormLinear(num_patches, token_mlp_dim),
            Swish(),
            nn.Dropout(dropout),
            WeightNormLinear(token_mlp_dim, num_patches),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        
        # 可选择使用MoE或标准MLP的Channel MLP
        if use_moe:
            self.channel_mlp = MoELayer(dim, num_experts=num_experts)
        else:
            self.channel_mlp = nn.Sequential(
                WeightNormLinear(dim, channel_mlp_dim),
                Swish(),
                nn.Dropout(dropout),
                WeightNormLinear(channel_mlp_dim, dim),
                nn.Dropout(dropout)
            )

    def forward(self, x):  # x shape: (B, num_patches, dim)
        # Token Mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)  # Shape: (B, dim, num_patches)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # Shape: (B, num_patches, dim)
        x = x + y

        # Channel Mixing
        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y
        return x

# 优化后的MLPMixerCustom
class MLPMixerCustom(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, token_mlp_dim, 
                 channel_mlp_dim, dropout=0., use_moe=False, num_experts=4, moe_layers=None):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # 指定哪些层使用MoE
        if moe_layers is None:
            moe_layers = [True] * depth if use_moe else [False] * depth
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, num_patches, token_mlp_dim, channel_mlp_dim, 
                      dropout=dropout, use_moe=moe_layers[i], num_experts=num_experts)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = WeightNormLinear(dim, num_classes)  # 输出层也使用权重归一化

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        skip_connection = x  # 保存初始特征作为跳跃连接
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = x + skip_connection  # 添加跳跃连接
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        x = self.head(x)
        return x
    
# 注册模型构建函数
def mlp_mixer_tiny_custom_builder(num_classes=100, **kwargs):
    return MLPMixerCustom(
    image_size=32,
    patch_size=4,
    dim=128,
    depth=8,
    num_classes=100,
    token_mlp_dim=256,
    channel_mlp_dim=512,
    use_moe=True,
    num_experts=4,
    moe_layers=[i % 2 == 0 for i in range(8)]  # 仅在偶数层使用MoE
)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.CIFAR100(root='./data', train=True,
                                 download=False, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False,
                                download=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 构建模型
model = mlp_mixer_tiny_custom_builder(num_classes=100)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

# 训练模型
num_epochs = 10
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
