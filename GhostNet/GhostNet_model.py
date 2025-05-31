import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ResNet import ResNet

# 模型定义
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential()
        if new_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1) if self.cheap_operation else torch.zeros_like(x1)
        out = torch.cat([x1, x2], dim=1)
        # if hasattr(self.cheap_operation, 'conv'):
        #      x2 = self.cheap_operation(x1)
        #      out = torch.cat([x1, x2], dim=1)
        # else:
        #      out = x1
        return out[:, :self.oup, :, :]
    
class GhostBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, ratio=2):
        super(GhostBasicBlock, self).__init__()
        self.ghost1 = GhostModule(in_planes, planes, kernel_size=3, ratio=ratio, dw_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ghost2 = GhostModule(planes, planes, kernel_size=3, ratio=ratio, dw_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.ghost1(x)))
        out = self.bn2(self.ghost2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 模型构建
def ghost_resnet20_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs={'ratio': ratio, **kwargs})

def ghost_resnet32_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs={'ratio': ratio, **kwargs})