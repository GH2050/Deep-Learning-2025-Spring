import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECABlock(nn.Module):
    """
    ECA 注意力模块：对输入特征进行 GAP 后，使用 1D 卷积学习通道权重。
    支持自适应确定卷积核大小或使用固定核大小。
    """
    def __init__(self, channels, gamma=2, b=1, adaptive_kernel=True, fixed_kernel_size=3):
        super(ECABlock, self).__init__()
        
        if adaptive_kernel:
            # 根据通道数自适应确定 1D 卷积核大小
            t = int(abs((math.log2(channels) + b) / gamma))
            k = t if t % 2 else t + 1
            k = max(k, 1)
        else:
            # 使用固定卷积核大小
            k = fixed_kernel_size
            
        self.kernel_size = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积：输入通道 1，输出通道 1，kernel_size=k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, 
                              padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B x C x H x W
        y = self.avg_pool(x)            # B x C x 1 x 1
        y = y.view(y.size(0), y.size(1))# B x C
        y = y.unsqueeze(1)             # B x 1 x C
        y = self.conv(y)               # B x 1 x C
        y = y.view(x.size(0), x.size(1), 1, 1)  # B x C x 1 x 1
        y = self.sigmoid(y)
        return x * y.expand_as(x)      # 通道乘以注意力权重


class BasicBlock(nn.Module):
    """ECANet 基本残差块，支持可选的 ECA 模块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_eca=False, adaptive_kernel=True, fixed_kernel_size=3, **kwargs):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # ECA 模块配置
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECABlock(planes, adaptive_kernel=adaptive_kernel, fixed_kernel_size=fixed_kernel_size)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 应用 ECA 注意力（如果启用）
        if self.use_eca:
            out = self.eca(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECANet(nn.Module):
    """
    ECANet 模型，基于 ResNet 架构并集成了 ECA 注意力模块
    支持控制是否使用 ECA 模块以及卷积核大小的计算方式
    """
    def __init__(self, block_type, num_blocks_list, num_classes=100, in_channels=3, 
                 use_eca=True, adaptive_kernel=True, fixed_kernel_size=3, block_kwargs=None):
        super(ECANet, self).__init__()
        
        self.in_planes = 64
        self.cifar_stem = True  # 适用于 CIFAR 数据集的设置
        self.use_eca = use_eca
        self.adaptive_kernel = adaptive_kernel
        self.fixed_kernel_size = fixed_kernel_size

        if block_kwargs is None:
            block_kwargs = {}

        # 将 ECA 相关参数添加到 block_kwargs
        block_kwargs.update({
            'use_eca': use_eca,
            'adaptive_kernel': adaptive_kernel,
            'fixed_kernel_size': fixed_kernel_size
        })

        if self.cifar_stem:
            self.in_planes = 16
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        planes_list = [16, 32, 64] if self.cifar_stem else [64, 128, 256, 512]
        strides = [1, 2, 2] if self.cifar_stem else [1, 2, 2, 2]
        
        current_planes = self.in_planes
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_list):
            stage_planes = planes_list[i]
            stage_stride = strides[i] if i > 0 or not self.cifar_stem else 1
            
            self.layers.append(self._make_layer(block_type, stage_planes, num_blocks, 
                                              stride=stage_stride, block_kwargs=block_kwargs))
            current_planes = stage_planes * block_type.expansion
            self.in_planes = current_planes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(current_planes, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride, block_kwargs):
        """构建多个残差块的阶段"""
        strides_list = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides_list:
            layers.append(block_type(self.in_planes, planes, s, **block_kwargs))
            self.in_planes = planes * block_type.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if not self.cifar_stem:
            out = self.maxpool(out)
        
        for layer_module in self.layers:
            out = layer_module(out)
            
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ecanet20_builder(num_classes=100, use_eca=True, adaptive_kernel=True, fixed_kernel_size=3, **kwargs):
    """
    构建 ECANet-20 模型
    
    Args:
        num_classes: 分类数量
        use_eca: 是否使用 ECA 模块
        adaptive_kernel: 是否使用自适应卷积核大小
        fixed_kernel_size: 固定卷积核大小（当 adaptive_kernel=False 时使用）
        **kwargs: 其他参数
    
    Returns:
        ECANet 模型实例
    """
    return ECANet(BasicBlock, [3, 3, 3], num_classes=num_classes, 
                  use_eca=use_eca, adaptive_kernel=adaptive_kernel, 
                  fixed_kernel_size=fixed_kernel_size, block_kwargs=kwargs)

# 对比试验
# 1.ECA + 自适应：
# model = ecanet20_builder(num_classes=100, use_eca=True, adaptive_kernel=True)
# 2.ECA + 固定 k=3：
# model = ecanet20_builder(num_classes=100, use_eca=True, adaptive_kernel=False, fixed_kernel_size=3)
# 3.无 ECA + 自适应：
# model = ecanet20_builder(num_classes=100, use_eca=False, adaptive_kernel=True)
# 4.无 ECA + 固定 k=3：
# model = ecanet20_builder(num_classes=100, use_eca=False, adaptive_kernel=False, fixed_kernel_size=3)
