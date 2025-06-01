import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, width_multiplier=1.0):
        super(ResNet, self).__init__()
        self.in_planes = int(16 * width_multiplier)
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.layer1 = self._make_layer(block, int(16 * width_multiplier), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32 * width_multiplier), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64 * width_multiplier), num_blocks[2], stride=2)
        
        self.linear = nn.Linear(int(64 * width_multiplier) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=100, width_multiplier=1.0):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, width_multiplier)

def resnet32(num_classes=100, width_multiplier=1.0):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, width_multiplier)

def resnet56(num_classes=100, width_multiplier=1.0):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, width_multiplier)

def resnet20_slim(num_classes=100):
    return resnet20(num_classes, width_multiplier=0.5)

def resnet32_slim(num_classes=100):
    return resnet32(num_classes, width_multiplier=0.5)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6