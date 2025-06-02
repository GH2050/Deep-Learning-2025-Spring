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


# ====== 改进版本: ResNet + 深度卷积 + 倒置瓶颈 ======
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
            expanded_planes,
            expanded_planes,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=expanded_planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expanded_planes)

        self.conv2 = nn.Conv2d(expanded_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        input_x = x

        out = F.relu(self.bn1(self.conv1(x)))  # expand
        out = F.relu(self.bn2(self.dwconv(out)))  # depthwise
        out = self.bn3(self.conv2(out))  # shrink

        out = self.shortcut(input_x) + self.drop_path(out)
        out = F.relu(out)
        return out


class ImprovedResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=100,
        width_multiplier=1.0,
        drop_path_rate=0.05,
    ):
        super().__init__()
        self.in_planes = int(16 * width_multiplier)

        # Traditional ResNet stem
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # 计算每个block的drop path rate
        total_blocks = sum(num_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # 构建层
        cur_idx = 0
        self.layer1 = self._make_layer(
            block,
            int(16 * width_multiplier),
            num_blocks[0],
            1,
            dp_rates[cur_idx : cur_idx + num_blocks[0]],
        )
        cur_idx += num_blocks[0]

        self.layer2 = self._make_layer(
            block,
            int(32 * width_multiplier),
            num_blocks[1],
            2,
            dp_rates[cur_idx : cur_idx + num_blocks[1]],
        )
        cur_idx += num_blocks[1]

        self.layer3 = self._make_layer(
            block,
            int(64 * width_multiplier),
            num_blocks[2],
            2,
            dp_rates[cur_idx : cur_idx + num_blocks[2]],
        )

        self.linear = nn.Linear(int(64 * width_multiplier), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dp_rates):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, (stride, dp_rate) in enumerate(zip(strides, dp_rates)):
            layers.append(block(self.in_planes, planes, stride, drop_path=dp_rate))
            self.in_planes = planes
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


# ====== CIFAR-100优化的ConvNeXt实现 ======
class ConvNeXtBlock_CIFAR(nn.Module):
    """针对CIFAR-100优化的ConvNeXt Block"""

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        expand_ratio=3,
        dropout_rate=0.1,
    ):
        super().__init__()

        expanded_dim = int(dim * expand_ratio)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)

        # MLP with dropout for regularization
        self.pwconv1 = nn.Conv2d(dim, expanded_dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pwconv2 = nn.Conv2d(expanded_dim, dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Layer Scale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.pwconv2(x)
        x = self.dropout2(x)

        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input_x + self.drop_path(x)
        return x


class ConvNeXt_CIFAR(nn.Module):
    """针对CIFAR-100优化的ConvNeXt"""

    def __init__(
        self,
        in_chans=3,
        num_classes=100,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        expand_ratio=3,
        dropout_rate=0.15,
        final_dropout=0.2,
    ):
        super().__init__()

        # Stem optimized for 32x32 images
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # 32->8
            LayerNorm2d(dims[0]),
            nn.Dropout2d(0.05),
        )

        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            # Stage blocks
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNeXtBlock_CIFAR(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        expand_ratio=expand_ratio,
                        dropout_rate=dropout_rate,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))

            # Downsampling (except last stage)
            if i < 3:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    nn.Dropout2d(0.1),
                )
                self.stages.append(downsample)

            cur += depths[i]

        # Final layers
        self.norm = LayerNorm2d(dims[-1])
        self.dropout = nn.Dropout(final_dropout)
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
        x = x.mean([-2, -1])  # Global average pooling
        x = self.dropout(x)
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


# Improved ResNet
def improved_resnet20_v2(num_classes=100, width_multiplier=1.0):
    """ResNet20 + Depthwise Conv + Inverted Bottleneck"""
    return ImprovedResNet(
        ImprovedBlock_v2, [3, 3, 3], num_classes, width_multiplier, drop_path_rate=0.05
    )


# CIFAR-100 Optimized ConvNeXt
def convnext_micro(num_classes=100, **kwargs):
    """超小版本 - 最适合CIFAR-100"""
    return ConvNeXt_CIFAR(
        depths=[2, 2, 6, 2],
        dims=[36, 72, 144, 288],
        drop_path_rate=0.08,
        expand_ratio=3.5,
        dropout_rate=0.05,
        final_dropout=0.1,
        num_classes=num_classes,
        **kwargs,
    )


def convnext_nano(num_classes=100, **kwargs):
    """纳米版本 - 平衡性能和过拟合"""
    return ConvNeXt_CIFAR(
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        drop_path_rate=0.2,
        expand_ratio=3,
        dropout_rate=0.15,
        final_dropout=0.2,
        num_classes=num_classes,
        **kwargs,
    )


def convnext_tiny_cifar(num_classes=100, **kwargs):
    """Tiny版本 - CIFAR优化"""
    return ConvNeXt_CIFAR(
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.25,
        expand_ratio=3,
        dropout_rate=0.15,
        final_dropout=0.25,
        num_classes=num_classes,
        **kwargs,
    )


def convnext_small_cifar(num_classes=100, **kwargs):
    """Small版本 - 需要更强正则化"""
    return ConvNeXt_CIFAR(
        depths=[2, 2, 18, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.3,
        expand_ratio=2.5,
        dropout_rate=0.2,
        final_dropout=0.3,
        num_classes=num_classes,
        **kwargs,
    )


# Legacy ConvNeXt (容易过拟合，不推荐用于CIFAR-100)
def convnext_tiny(num_classes=100, **kwargs):
    """原始ConvNeXt-Tiny (不推荐用于CIFAR-100)"""
    from warnings import warn

    warn(
        "convnext_tiny容易在CIFAR-100上过拟合，建议使用convnext_micro或convnext_nano",
        UserWarning,
    )

    class ConvNeXtBlock_Original(nn.Module):
        def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = LayerNorm2d(dim, eps=1e-6)
            self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

            self.gamma = (
                nn.Parameter(
                    layer_scale_init_value * torch.ones((dim)), requires_grad=True
                )
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

    class ConvNeXt_Original(nn.Module):
        def __init__(
            self,
            depths=[2, 2, 6, 2],
            dims=[48, 96, 192, 384],
            drop_path_rate=0.0,
            num_classes=100,
            **kwargs,
        ):
            super().__init__()

            self.stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )

            self.stages = nn.ModuleList()
            dp_rates = [
                x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
            ]
            cur = 0

            for i in range(4):
                stage_blocks = []
                for j in range(depths[i]):
                    stage_blocks.append(
                        ConvNeXtBlock_Original(
                            dim=dims[i],
                            drop_path=dp_rates[cur + j],
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

    return ConvNeXt_Original(num_classes=num_classes, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def get_training_config(model_name):
    """为不同模型返回推荐的训练配置"""
    configs = {
        # ConvNeXt系列 - 改进配置
        "convnext_micro": {
            "lr": 0.004,
            "weight_decay": 0.005,
            "optimizer": "adamw",
            "epochs": 300,  # 延长训练
            "warmup_epochs": 10,
            "label_smoothing": 0.03,
        },
        "convnext_nano": {
            "lr": 0.003,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "epochs": 350,
            "warmup_epochs": 15,
            "label_smoothing": 0.05,
        },
        "convnext_tiny_cifar": {
            "lr": 0.002,
            "weight_decay": 0.02,
            "optimizer": "adamw",
            "epochs": 400,
            "warmup_epochs": 15,
            "label_smoothing": 0.08,
        },
        "convnext_small_cifar": {
            "lr": 0.001,
            "weight_decay": 0.03,
            "optimizer": "adamw",
            "epochs": 400,
            "warmup_epochs": 20,
            "label_smoothing": 0.1,
        },
        # ResNet系列
        "resnet20": {
            "lr": 0.1,
            "weight_decay": 5e-4,
            "optimizer": "sgd",
            "epochs": 200,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
        },
        "resnet32": {
            "lr": 0.1,
            "weight_decay": 5e-4,
            "optimizer": "sgd",
            "epochs": 200,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
        },
        "resnet56": {
            "lr": 0.1,
            "weight_decay": 5e-4,
            "optimizer": "sgd",
            "epochs": 250,
            "warmup_epochs": 5,
            "label_smoothing": 0.0,
        },
        "resnet20_slim": {
            "lr": 0.1,
            "weight_decay": 1e-3,
            "optimizer": "sgd",
            "epochs": 200,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
        },
        "resnet32_slim": {
            "lr": 0.1,
            "weight_decay": 1e-3,
            "optimizer": "sgd",
            "epochs": 200,
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
        },
        "improved_resnet20_v2": {
            "lr": 0.05,
            "weight_decay": 1e-3,
            "optimizer": "sgd",
            "epochs": 250,
            "warmup_epochs": 5,
            "label_smoothing": 0.05,
        },
        # 原版ConvNeXt (不推荐)
        "convnext_tiny": {
            "lr": 0.0005,
            "weight_decay": 0.05,
            "optimizer": "adamw",
            "epochs": 400,
            "warmup_epochs": 20,
            "label_smoothing": 0.2,
        },
    }

    return configs.get(model_name, configs["resnet20"])  # 默认使用ResNet配置


# ====== 测试代码 ======
if __name__ == "__main__":
    print("CIFAR-100 Optimized Models:")
    print("=" * 80)

    models_to_test = [
        ("resnet20", resnet20()),
        ("resnet20_slim", resnet20_slim()),
        ("improved_resnet20_v2", improved_resnet20_v2()),
        ("convnext_micro", convnext_micro()),
        ("convnext_nano", convnext_nano()),
        ("convnext_tiny_cifar", convnext_tiny_cifar()),
        ("convnext_tiny_original", convnext_tiny()),
    ]

    print(
        f"{'Model':<25} {'Parameters':<12} {'Output Shape':<15} {'Recommended':<12} {'Status'}"
    )
    print("-" * 80)

    x = torch.randn(2, 3, 32, 32)

    recommendations = {
        "resnet20": "Baseline",
        "resnet20_slim": "Fast",
        "improved_resnet20_v2": "Good",
        "convnext_micro": "Best",
        "convnext_nano": "Excellent",
        "convnext_tiny_cifar": "Good",
        "convnext_tiny_original": "Avoid",
    }

    for name, model in models_to_test:
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
            params = count_parameters(model)
            rec = recommendations.get(name, "Unknown")
            print(
                f"{name:<25} {params:<12.2f}M {str(output.shape):<15} {rec:<12} {'✓'}"
            )
        except Exception as e:
            print(f"{name:<25} {'ERROR':<12} {'':<15} {'':<12} {'✗'}")

    print("-" * 80)
    print("\nTraining Recommendations:")
    print("- convnext_micro: 最佳选择，参数少，不易过拟合")
    print("- convnext_nano: 性能优秀，平衡过拟合风险")
    print("- improved_resnet20_v2: ResNet改进版，效率高")
    print("- 避免使用 convnext_tiny_original: 容易过拟合")
    print("\n使用 get_training_config(model_name) 获取推荐的训练参数")
