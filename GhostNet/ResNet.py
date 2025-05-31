import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=100, in_channels=3, block_kwargs=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.cifar_stem = True

        if block_kwargs is None:
            block_kwargs = {}

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
            
            self.layers.append(self._make_layer(block_type, stage_planes, num_blocks, stride=stage_stride, block_kwargs=block_kwargs))
            current_planes = stage_planes * block_type.expansion
            self.in_planes = current_planes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(current_planes, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride, block_kwargs):
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
