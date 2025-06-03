
### MLP（多层感知机）基础介绍

MLP（Multi-Layer Perceptron）是一种经典的人工神经网络结构，由输入层、隐藏层和输出层组成，各层神经元之间全连接。传统MLP存在以下特点：

1. **结构简单但表达能力有限**：通过堆叠全连接层可以拟合任意复杂函数，但参数量随层数增加呈指数级增长，易过拟合。

2. **训练难题**：深层MLP会面临梯度消失/爆炸问题，且对于图像等结构化数据，缺乏对空间局部特征的捕捉能力。

针对传统MLP的不足，从架构设计、参数优化和激活函数等维度进行了深度改进：

#### 1. **权重归一化（Weight Normalization）**

```python
class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(out_features))
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        # 将权重分解为方向(v)和大小(g)
        ...
        
    def forward(self, x):
        weight = F.normalize(self.weight_v, dim=1) * self.weight_g.unsqueeze(1)
        return F.linear(x, weight, self.bias)
```

**作用**：
- 将权重参数分解为方向向量(`v`)和标量大小(`g`)，通过归一化权重方向缓解梯度不稳定问题。
- 相比Batch Normalization，不依赖批量统计信息，更适合小批量训练和RNN等动态网络。

#### 2. **Swish自适应激活函数**

```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

**优势**：
- 平滑的非线性激活，兼具ReLU和Sigmoid的优点，在深层网络中表现优于ReLU。
- 无界性避免了梯度饱和，同时具有自门控特性，有助于捕捉复杂模式。

#### 3. **混合专家模型（MoE, Mixture of Experts）**

```python
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, ...):
        self.experts = nn.ModuleList([Expert(...) for _ in range(num_experts)])
        self.gating = GatingNetwork(dim, num_experts)
        ...
        
    def forward(self, x):
        gates = self.gating(x_flat)  # 计算门控权重
        top_values, top_indices = torch.topk(gates, 2)  # 选择Top-2专家
        # 动态路由至多个专家
        for i in range(self.num_experts):
            mask = (top_indices == i).float()
            expert_outputs += mask * self.experts[i](x_flat * mask)
        return expert_outputs
```

**创新点**：
- **动态计算路径**：通过门控网络为每个输入样本分配不同的专家组合，实现模型容量与计算效率的平衡。
- **细粒度特征学习**：不同专家可专注于不同类型的特征（如纹理、形状），提升对CIFAR-100的分类能力。
- **稀疏激活**：每个样本仅激活部分专家，降低整体计算量，同时避免过拟合。

#### 4. **MixerBlock结构优化**

```python
class MixerBlock(nn.Module):
    def __init__(self, ..., use_moe=False):
        # Token Mixing: 跨patch交互
        self.token_mlp = nn.Sequential(
            WeightNormLinear(num_patches, token_mlp_dim),
            Swish(), ...
        )
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.channel_mlp = MoELayer(...) if use_moe else nn.Sequential(...)
```

**设计特点**：
- **分离Token和Channel混合**：借鉴MLP-Mixer思想，分别处理空间（patch间）和通道（特征维度）信息。
- **轻量化**：减少了后续网络需要处理的序列长度（例如，32×32 图像分割为 8×8=64 个 patch），降低计算复杂度。


