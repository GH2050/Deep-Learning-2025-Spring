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

# ConvNeXt Implementation
class LayerNorm2d(nn.Module):
    """2D LayerNorm for ConvNeXt"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block as described in the paper"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise convolution (7x7 kernel)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Layer normalization
        self.norm = LayerNorm2d(dim, eps=1e-6)
        # Pointwise/1x1 convolutions, in MLP style
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # shrink
        
        # Layer scale (gamma parameter)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)        # Depthwise conv
        x = self.norm(x)          # LayerNorm
        x = self.pwconv1(x)       # 1x1 conv expand
        x = self.act(x)           # GELU
        x = self.pwconv2(x)       # 1x1 conv shrink
        
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        x = input_x + self.drop_path(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNeXt(nn.Module):
    """ConvNeXt adapted for CIFAR-100"""
    def __init__(self, in_chans=3, num_classes=100, 
                 depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        # Stem: "patchify" layer for CIFAR-100 (smaller patch size)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # 32->8 for CIFAR
            LayerNorm2d(dims[0])
        )
        
        self.depths = depths
        self.stages = nn.ModuleList()  # 4 feature resolution stages
        
        # Build stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            # Each stage: blocks + optional downsampling
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ))
            self.stages.append(nn.Sequential(*stage_blocks))
            
            # Downsampling layer between stages (except the last)
            if i < 3:
                self.stages.append(nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))
            
            cur += depths[i]
        
        # Final norm and classifier
        self.norm = LayerNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.head(x)
        return x

# Model factory functions for different ConvNeXt sizes

def convnext_tiny(num_classes=100, **kwargs):
    """ConvNeXt-T: Tiny model adapted for CIFAR-100"""
    # Scaled down from original for CIFAR-100
    return ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], 
                   num_classes=num_classes, **kwargs)

def convnext_small(num_classes=100, **kwargs):
    """ConvNeXt-S: Small model adapted for CIFAR-100"""
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[48, 96, 192, 384], 
                   num_classes=num_classes, **kwargs)

def convnext_base(num_classes=100, **kwargs):
    """ConvNeXt-B: Base model adapted for CIFAR-100"""
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[64, 128, 256, 512], 
                   num_classes=num_classes, **kwargs)

def convnext_large(num_classes=100, **kwargs):
    """ConvNeXt-L: Large model adapted for CIFAR-100"""
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[96, 192, 384, 768], 
                   num_classes=num_classes, **kwargs)

# Original ResNet functions
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

# Test function
if __name__ == '__main__':
    # Test all models
    models_to_test = [
        ('resnet20', resnet20()),
        ('resnet32', resnet32()),
        ('resnet56', resnet56()),
        ('resnet20_slim', resnet20_slim()),
        ('resnet32_slim', resnet32_slim()),
        ('convnext_tiny', convnext_tiny()),
        ('convnext_small', convnext_small()),
        ('convnext_base', convnext_base()),
        ('convnext_large', convnext_large()),
    ]
    
    print("Model Testing for CIFAR-100 (32x32x3 input):")
    print("-" * 60)
    print(f"{'Model':<15} {'Parameters':<12} {'Output Shape':<15} {'Status'}")
    print("-" * 60)
    
    x = torch.randn(2, 3, 32, 32)  # CIFAR-100 input size
    
    for name, model in models_to_test:
        try:
            with torch.no_grad():
                output = model(x)
            params = count_parameters(model)
            print(f"{name:<15} {params:<12.2f}M {str(output.shape):<15} {'✓'}")
        except Exception as e:
            print(f"{name:<15} {'ERROR':<12} {'':<15} {'✗'} {str(e)[:30]}...")
    
    print("-" * 60)