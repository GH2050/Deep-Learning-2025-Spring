import torch
import torch.nn as nn
import torch.nn.functional as F

#扰动模块
class ChannelPerturbation(nn.Module):
    def __init__(self, num_channels, perturb_strength=0.1):
        super(ChannelPerturbation, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1, num_channels, 1, 1) * perturb_strength)

    def forward(self, x):
        noise = torch.tanh(self.alpha)
        return x + noise


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_perturb=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.use_perturb = use_perturb

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.perturb1 = ChannelPerturbation(planes) if use_perturb else nn.Identity()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.perturb2 = ChannelPerturbation(planes) if use_perturb else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.perturb1(self.bn1(self.conv1(x))))
        out = self.perturb2(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=100, in_channels=3, block_kwargs=None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        if block_kwargs is None:
            block_kwargs = {}

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        planes_list = [16, 32, 64]
        strides = [1, 2, 2]

        current_planes = self.in_planes
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_list):
            stage_planes = planes_list[i]
            stage_stride = strides[i]
            
            self.layers.append(self._make_layer(block_type, stage_planes, num_blocks, stride=stage_stride, block_kwargs=block_kwargs))
            current_planes = stage_planes * block_type.expansion
            self.in_planes = current_planes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(current_planes, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride, block_kwargs):
        strides_list = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides_list:
            layers.append(block_type(self.in_planes, planes, s, **block_kwargs))
            self.in_planes = planes * block_type.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer_module in self.layers:
            out = layer_module(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def newmodel_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs=kwargs)
