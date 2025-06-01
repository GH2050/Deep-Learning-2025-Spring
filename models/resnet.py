# models/resnet.py
import torch
import torch.nn as nn
from .eca import ECABlock  # 导入 ECA 模块

class BasicBlock(nn.Module):
    """ResNet 基本残差块（两个 3x3 卷积）"""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, use_eca=False):
        super(BasicBlock, self).__init__()
        # 第一个 3x3 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个 3x3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 如果尺寸变化，需要下采样的 shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        # 是否使用 ECA 注意力
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECABlock(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 应用 ECA 注意力（可选）
        if self.use_eca:
            out = self.eca(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet 网络（以 ResNet-20 为例），支持可选 ECA 模块"""
    def __init__(self, block, num_blocks, num_classes=100, use_eca=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.use_eca = use_eca
        # 初始的 3x3 卷积
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 三个阶段，每个阶段包含 num_blocks 个残差块
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """构建多个残差块的阶段"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, use_eca=self.use_eca))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)            # 全局平均池化至 1×1
        out = out.view(out.size(0), -1)    # 拉平
        out = self.fc(out)                 # 全连接输出
        return out
