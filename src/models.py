import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
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

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(
            block, int(16 * width_multiplier), num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, int(32 * width_multiplier), num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, int(64 * width_multiplier), num_blocks[2], stride=2
        )

        self.linear = nn.Linear(
            int(64 * width_multiplier) * block.expansion, num_classes
        )

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


# ====== ConvNeXt Components ======
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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ====== 改进版本 1: ResNet + 大卷积核 ======
class ImprovedBlock_v1(nn.Module):
    """ResNet + 7x7 kernel"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ====== 改进版本 2: ResNet + 深度卷积 + 倒置瓶颈 ======
class ImprovedBlock_v2(nn.Module):
    """ResNet + Depthwise Conv + Inverted Bottleneck"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        expand_ratio = 4
        expanded_planes = in_planes * expand_ratio

        # Inverted bottleneck: expand -> depthwise -> shrink
        self.conv1 = nn.Conv2d(in_planes, expanded_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_planes)
        
        self.dwconv = nn.Conv2d(
            expanded_planes, expanded_planes, kernel_size=7, 
            stride=stride, padding=3, groups=expanded_planes, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expanded_planes)
        
        self.conv2 = nn.Conv2d(expanded_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # Drop Path (虽然这个版本主要用BatchNorm+ReLU，但也支持drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        input_x = x
        
        out = F.relu(self.bn1(self.conv1(x)))      # expand
        out = F.relu(self.bn2(self.dwconv(out)))  # depthwise
        out = self.bn3(self.conv2(out))           # shrink
        
        # Apply drop path and residual connection
        out = self.shortcut(input_x) + self.drop_path(out)
        out = F.relu(out)
        return out


# ====== 改进版本 3: ResNet + ConvNeXt 特性 ======
class ImprovedBlock_v3(nn.Module):
    """ResNet + LayerNorm + GELU + Layer Scale + Drop Path"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        expand_ratio = 4
        expanded_planes = in_planes * expand_ratio
        layer_scale_init_value = 1e-6

        # ConvNeXt-style: depthwise -> norm -> expand -> gelu -> shrink
        self.dwconv = nn.Conv2d(
            in_planes, in_planes, kernel_size=7, 
            stride=stride, padding=3, groups=in_planes, bias=False
        )
        self.norm1 = LayerNorm2d(in_planes)
        
        self.conv1 = nn.Conv2d(in_planes, expanded_planes, kernel_size=1, bias=False)
        self.act = nn.GELU()  # GELU instead of ReLU
        self.conv2 = nn.Conv2d(expanded_planes, planes, kernel_size=1, bias=False)
        
        # Layer Scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(planes), requires_grad=True
        )
        
        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(planes),
            )

    def forward(self, x):
        input_x = x
        
        out = self.dwconv(x)      # depthwise conv
        out = self.norm1(out)     # LayerNorm
        out = self.conv1(out)     # expand
        out = self.act(out)       # GELU (only one activation!)
        out = self.conv2(out)     # shrink
        
        # Layer scale
        out = self.gamma.view(1, -1, 1, 1) * out
        
        # Residual connection with drop path
        out = self.shortcut(input_x) + self.drop_path(out)
        
        return out


# ====== 改进版本 4: 最接近 ConvNeXt 的 ResNet ======
class ImprovedBlock_v4(nn.Module):
    """Most ConvNeXt-like ResNet Block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        expand_ratio = 4
        layer_scale_init_value = 1e-6

        # 处理步长下采样: 当stride>1时，使用普通卷积；否则使用深度卷积
        if stride > 1:
            # 使用普通卷积处理下采样，然后调整通道数
            self.dwconv = nn.Conv2d(
                in_planes, planes, kernel_size=7, 
                stride=stride, padding=3, bias=False
            )
            self.norm = LayerNorm2d(planes)
            self.conv1 = nn.Conv2d(planes, expand_ratio * planes, kernel_size=1)
            self.conv2 = nn.Conv2d(expand_ratio * planes, planes, kernel_size=1)
        else:
            # 使用深度卷积
            self.dwconv = nn.Conv2d(
                in_planes, in_planes, kernel_size=7, 
                stride=1, padding=3, groups=in_planes, bias=False
            )
            self.norm = LayerNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, expand_ratio * in_planes, kernel_size=1)
            self.conv2 = nn.Conv2d(expand_ratio * in_planes, planes, kernel_size=1)
            
        self.act = nn.GELU()
        
        # Layer Scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(planes), requires_grad=True
        )
        
        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(planes),
            )

    def forward(self, x):
        input_x = x
        
        # ConvNeXt-style forward
        out = self.dwconv(x)      # depthwise or regular conv
        out = self.norm(out)      # LayerNorm
        out = self.conv1(out)     # expand
        out = self.act(out)       # GELU
        out = self.conv2(out)     # shrink
        
        # Layer scale
        out = self.gamma.view(1, -1, 1, 1) * out
        
        # Residual connection
        out = self.shortcut(input_x) + self.drop_path(out)
        
        return out


# ====== 通用的改进 ResNet 架构 ======
class ImprovedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, width_multiplier=1.0, 
                 drop_path_rate=0.1, use_stem_v3=False):
        super().__init__()
        self.in_planes = int(16 * width_multiplier)
        
        # Stem 层
        if use_stem_v3:
            # ConvNeXt-style patchify stem
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.in_planes, kernel_size=4, stride=4, bias=False),
                LayerNorm2d(self.in_planes)
            )
        else:
            # Traditional ResNet stem
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # 计算每个block的drop path rate
        total_blocks = sum(num_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # 构建层
        cur_idx = 0
        self.layer1 = self._make_layer(
            block, int(16 * width_multiplier), num_blocks[0], 1,
            dp_rates[cur_idx:cur_idx + num_blocks[0]]
        )
        cur_idx += num_blocks[0]
        
        self.layer2 = self._make_layer(
            block, int(32 * width_multiplier), num_blocks[1], 2,
            dp_rates[cur_idx:cur_idx + num_blocks[1]]
        )
        cur_idx += num_blocks[1]
        
        self.layer3 = self._make_layer(
            block, int(64 * width_multiplier), num_blocks[2], 2,
            dp_rates[cur_idx:cur_idx + num_blocks[2]]
        )
        
        # 最终层
        final_dim = int(64 * width_multiplier)
        if use_stem_v3:
            self.norm = LayerNorm2d(final_dim)
        self.linear = nn.Linear(final_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dp_rates):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, (stride, dp_rate) in enumerate(zip(strides, dp_rates)):
            # 统一传递 drop_path 参数，所有block都支持了
            layers.append(block(self.in_planes, planes, stride, drop_path=dp_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self, 'stem'):
            out = self.stem(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        if hasattr(self, 'norm'):
            out = self.norm(out)
            
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ====== ConvNeXt 实现 ======
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input_x + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt adapted for CIFAR-100"""

    def __init__(
        self,
        in_chans=3,
        num_classes=100,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))

            if i < 3:
                self.stages.append(
                    nn.Sequential(
                        LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                )

            cur += depths[i]

        self.norm = LayerNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x


# ====== 模型工厂函数 ======

# Original ResNet
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

# Improved ResNet versions
def improved_resnet20_v1(num_classes=100, width_multiplier=1.0):
    """ResNet20 + 7x7 kernel"""
    return ImprovedResNet(ImprovedBlock_v1, [3, 3, 3], num_classes, width_multiplier, 
                         drop_path_rate=0.0, use_stem_v3=False)

def improved_resnet20_v2(num_classes=100, width_multiplier=1.0):
    """ResNet20 + Depthwise Conv + Inverted Bottleneck"""
    return ImprovedResNet(ImprovedBlock_v2, [3, 3, 3], num_classes, width_multiplier, 
                         drop_path_rate=0.05, use_stem_v3=False)

def improved_resnet20_v3(num_classes=100, width_multiplier=1.0, drop_path_rate=0.1):
    """ResNet20 + ConvNeXt features"""
    return ImprovedResNet(ImprovedBlock_v3, [3, 3, 3], num_classes, width_multiplier, 
                         drop_path_rate=drop_path_rate, use_stem_v3=True)

def improved_resnet20_v4(num_classes=100, width_multiplier=1.0, drop_path_rate=0.1):
    """ResNet20 most ConvNeXt-like"""
    return ImprovedResNet(ImprovedBlock_v4, [3, 3, 3], num_classes, width_multiplier, 
                         drop_path_rate=drop_path_rate, use_stem_v3=True)

# ConvNeXt models
def convnext_tiny(num_classes=100, **kwargs):
    return ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], num_classes=num_classes, **kwargs)

def convnext_small(num_classes=100, **kwargs):
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[48, 96, 192, 384], num_classes=num_classes, **kwargs)

def convnext_base(num_classes=100, **kwargs):
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[64, 128, 256, 512], num_classes=num_classes, **kwargs)

def convnext_large(num_classes=100, **kwargs):
    return ConvNeXt(depths=[2, 2, 18, 2], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# ====== 测试代码 ======
if __name__ == '__main__':
    models_to_test = [
        ('resnet20', resnet20()),
        ('improved_resnet20_v1', improved_resnet20_v1()),
        ('improved_resnet20_v2', improved_resnet20_v2()),
        ('improved_resnet20_v3', improved_resnet20_v3()),
        ('improved_resnet20_v4', improved_resnet20_v4()),
        ('convnext_tiny', convnext_tiny()),
    ]
    
    print("Model Comparison for CIFAR-100:")
    print("-" * 90)
    print(f"{'Model':<20} {'Parameters':<12} {'Output Shape':<15} {'Key Features':<30} {'Status'}")
    print("-" * 90)
    
    x = torch.randn(2, 3, 32, 32)
    
    features_map = {
        'resnet20': 'Baseline ResNet',
        'improved_resnet20_v1': '7x7 kernel',
        'improved_resnet20_v2': 'DWConv + Inverted Bottleneck',
        'improved_resnet20_v3': 'LayerNorm + GELU + DropPath',
        'improved_resnet20_v4': 'Most ConvNeXt-like',
        'convnext_tiny': 'Full ConvNeXt'
    }
    
    for name, model in models_to_test:
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
            params = count_parameters(model)
            features = features_map.get(name, 'Unknown')
            print(f"{name:<20} {params:<12.2f}M {str(output.shape):<15} {features:<30} {'✓'}")
        except Exception as e:
            print(f"{name:<20} {'ERROR':<12} {'':<15} {'':<30} {'✗'} {str(e)[:30]}...")
    
    print("-" * 90)