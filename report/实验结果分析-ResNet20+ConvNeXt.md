参考文献：A convnet for the 2020s（https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf）

ConvNeXt 是 Facebook AI Research 在 2022 年提出的一项重要工作，其核心思想是通过系统性地重新审视卷积神经网络的设计空间，证明传统卷积网络在适当改进后仍能与现代 Vision Transformer 竞争。

ConvNeXt 的核心思想是以 ResNet 为起点，逐步引入 Vision Transformer 中有效的结构和训练策略，打造一个纯卷积网络。技术上，ConvNeXt 采用了更强的数据增强和优化器（如 AdamW、Mixup、CutMix、Stochastic Depth），将网络初始的 Stem 换成 4×4 步长的卷积实现 Patchify，采用更合理的 stage 比例和通道加宽，主干结构用深度可分离卷积和反向瓶颈设计，并将大核（如 7×7）用于 depthwise convolution。同时，激活函数采用 GELU，每个残差块仅保留一个激活和归一化层，归一化方式采用 LayerNorm，分离下采样层并在分辨率变化处增加归一化。


复现过程中我们发现即使将论文中的 ConvNeXt 架构直接适配到 CIFAR-100 数据集，仍然存在容易过拟合的问题。这主要源于 CIFAR-100 相比 ImageNet 的数据规模较小（5万训练样本 vs 120万训练样本），而 ConvNeXt 的原始设计是为大规模数据集优化的。因此我们采取了选择性借鉴关键技术的策略。

### 借鉴 ConvNeXt

**倒置瓶颈结构 (Inverted Bottleneck)**

倒置瓶颈最初由 MobileNetV2 提出，其设计与传统瓶颈结构相反。传统瓶颈是"压缩-处理-恢复"，而倒置瓶颈是"扩展-处理-压缩"。

```
# 传统瓶颈：64 → 16 → 16 → 64 (先压缩)
# 倒置瓶颈：64 → 256 → 256 → 64 (先扩展)
expand_ratio = 4
expanded_planes = in_planes * expand_ratio

self.conv1 = nn.Conv2d(in_planes, expanded_planes, kernel_size=1)  # 扩展
# 中间处理层
self.conv2 = nn.Conv2d(expanded_planes, planes, kernel_size=1)     # 压缩
```

**深度可分离卷积 (Depthwise Separable Convolution)**

深度可分离卷积将标准卷积分解为两个独立的操作：深度卷积（每个通道独立处理）和逐点卷积（通道间信息融合）。

可以减少参数量和计算量，将空间特征提取与通道信息融合解耦，与 Transformer 中的注意力机制在概念上相似。

我们的实现：

```
self.dwconv = nn.Conv2d(
    expanded_planes, expanded_planes, 
    kernel_size=7, stride=stride, padding=3,
    groups=expanded_planes,  # groups=channels 实现深度卷积
    bias=False
)
```

**DropPath 正则化技术**

DropPath（随机深度）是一种路径级别的正则化技术，它随机删除整个残差分支，而不是像 Dropout 那样删除单个神经元。

```
def forward(self, x):
    if self.drop_prob == 0. or not self.training:
        return x
    
    keep_prob = 1 - self.drop_prob
    # 只在 batch 维度随机，保持其他维度一致
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    
    return x.div(keep_prob) * random_tensor
```

**我们模型的技术借鉴与适配**

架构设计对比

| 技术组件 | ConvNeXt 原版 | improved_resnet20_convnext | 适配说明 |
|----------|---------------|-------------------|----------|
| **瓶颈结构** | 倒置瓶颈 (4×扩展) | 倒置瓶颈 (4×扩展) | 完全采用 |
| **卷积核大小** | 7×7 深度卷积 | 7×7 深度卷积 | 完全采用 |
| **正则化** | DropPath | DropPath | 完全采用 |
| **归一化** | LayerNorm | BatchNorm | CIFAR 适配 |
| **激活函数** | GELU | ReLU | 保持简洁 |


### 实验数据：

resnet_20:
```
"best_test_accuracy_top1": 66.62,
"final_test_accuracy_top1": 66.23,
```
![](..\logs\resnet_20\20250603-014122\training_curves.png)

improved_resnet20_convnext: 

```
"best_test_accuracy_top1": 72.33,
"final_test_accuracy_top1": 72.15,
```

![](..\logs\improved_resnet20_convnext\20250603-033619\training_curves.png)

### 消融实验设计

设计思路：
- 每次只改变一个技术组件
- 逐步简化：从完整版本逐步移除关键技术
- 性能量化：通过准确率差异量化每个组件的贡献

（数据待补充）