import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from typing import Optional, List

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

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

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(GhostBottleneck, self).__init__()
        hidden_dim = in_planes
        
        self.ghost1 = GhostModule(in_planes, hidden_dim, kernel_size=1, relu=True)
        
        if stride > 1:
            self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)
        
        self.ghost2 = GhostModule(hidden_dim, planes, kernel_size=1, relu=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(planes),
            )
        
        self.stride = stride

    def forward(self, x):
        residual = x
        
        x = self.ghost1(x)
        
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
            
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        x = input + x
        return x

# SegNeXt MSCA模块
class MSCA(nn.Module):
    def __init__(self, dim, kernel_sizes=[7, 11], scale_factor=4):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.scales = nn.ModuleList()
        self.scales.append(nn.Identity())
        
        for ks in kernel_sizes:
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=(1, ks), padding=(0, ks//2), groups=dim),
                    nn.Conv2d(dim, dim, kernel_size=(ks, 1), padding=(ks//2, 0), groups=dim)
                )
            )
        
        self.conv_channel_mixer = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        base_feat = self.conv0(x)
        
        scale_outputs = []
        for scale_conv in self.scales:
            scale_outputs.append(scale_conv(base_feat))
        
        summed_feats = torch.sum(torch.stack(scale_outputs), dim=0)
        attention_map = self.conv_channel_mixer(summed_feats)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class MSCABlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.msca = MSCA(dim)
        self.norm1 = nn.BatchNorm2d(dim)
        
        self.conv1 = nn.Conv2d(dim, dim * 4, 1)
        self.conv2 = nn.Conv2d(dim * 4, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim * 4)
        self.act = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.msca(x)
        x = x + residual
        
        residual = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x + residual
        
        return x

# LSK模块
class LSKBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim//2, dim, 1)
        self.conv3 = nn.Conv2d(dim//2, dim, 1)
        
    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = attn1 + attn2
        
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = avg_attn + max_attn
        
        sig = torch.sigmoid(agg)
        return x * sig

# CSP块
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv3 = nn.Conv2d(2 * hidden_channels, out_channels, 1)
        
        module_list = []
        for _ in range(num_blocks):
            module_list.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.blocks = nn.Sequential(*module_list)
        
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.blocks(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        return self.conv3(x)

# ResNeSt分裂注意力
class SplitAttention(nn.Module):
    def __init__(self, channels, radix=2):
        super().__init__()
        self.radix = radix
        self.channels = channels
        self.inter_channels = max(channels * radix // 4, 32)
        
        self.conv = nn.Conv2d(channels, self.inter_channels, 1, groups=1)
        self.bn = nn.BatchNorm2d(self.inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.inter_channels, channels * radix, 1, groups=1)
        self.rsoftmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch, channel = x.size()[:2]
        splited = torch.split(x, channel // self.radix, dim=1)
        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.conv(gap)
        gap = self.bn(gap)
        gap = self.relu(gap)
        
        atten = self.fc1(gap)
        atten = atten.view(batch, self.radix, channel // self.radix)
        atten = self.rsoftmax(atten)
        atten = atten.view(batch, channel, 1, 1)
        
        attens = torch.split(atten, channel // self.radix, dim=1)
        out = sum([att * split for att, split in zip(attens, splited)])
        return out

# MLP-Mixer块
class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
        )
        
        self.token_mlp = MLPBlock(num_patches, token_dim, dropout)
        
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
        )
        
        self.channel_mlp = MLPBlock(dim, channel_dim, dropout)
        
    def forward(self, x):
        # Token mixing: operate on the token dimension
        residual = x
        x = self.token_mix[0](x)  # LayerNorm
        x = x.transpose(1, 2)  # (B, num_patches, dim) -> (B, dim, num_patches)
        x = self.token_mlp(x)
        x = x.transpose(1, 2)  # (B, dim, num_patches) -> (B, num_patches, dim)
        x = x + residual
        
        # Channel mixing: operate on the channel dimension
        residual = x
        x = self.channel_mix[0](x)  # LayerNorm
        x = self.channel_mlp(x)
        x = x + residual
        
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class ConvNeXtResNet(nn.Module):
    def __init__(self, depths=[3, 3, 3], dims=[16, 32, 64], num_classes=100):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )
                self.stages.append(downsample)
                
        self.norm = LayerNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

# MSCAN编码器
class MSCANEncoder(nn.Module):
    def __init__(self, dims=[32, 64, 128], depths=[3, 4, 6], num_classes=100):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 7, 4, 3),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[MSCABlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], 3, 2, 1),
                    nn.BatchNorm2d(dims[i+1])
                )
                self.stages.append(downsample)
        
        self.norm = nn.BatchNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

# MLP-Mixer实现
class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=256, num_classes=100, depth=8, token_dim=256, channel_dim=512):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dim = dim
        
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        
        self.mixer_layers = nn.ModuleList([
            MixerBlock(dim, self.num_patches, token_dim, channel_dim)
            for _ in range(depth)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # Patch embedding: (B, 3, H, W) -> (B, dim, H/P, W/P) -> (B, num_patches, dim)
        x = self.patch_embed(x)  # (B, dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        for mixer in self.mixer_layers:
            x = mixer(x)
            
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.head(x)

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def eca_resnet20():
    return ResNet(ECABasicBlock, [3, 3, 3])

def eca_resnet32():
    return ResNet(ECABasicBlock, [5, 5, 5])

def ghost_resnet20():
    return ResNet(GhostBottleneck, [3, 3, 3])

def ghost_resnet32():
    return ResNet(GhostBottleneck, [5, 5, 5])

def convnext_tiny():
    return ConvNeXtResNet([2, 2, 6], [16, 32, 64])

def segnext_mscan_tiny():
    return MSCANEncoder([32, 64, 128], [2, 2, 4])

def mlp_mixer_tiny():
    return MLPMixer(32, 4, 256, 100, 6, 256, 512)

# 通过timm库创建预训练模型的包装函数
def create_timm_model(model_name: str, num_classes: int = 100, pretrained: bool = True):
    """创建timm预训练模型"""
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        print(f"创建timm模型 {model_name} 失败: {e}")
        return None

def convnext_tiny_timm():
    return create_timm_model('convnext_tiny', 100, True)

def coatnet_0():
    return create_timm_model('coatnet_0_rw_224', 100, True)

def cspresnet50():
    return create_timm_model('cspresnet50', 100, True)

def ghostnet_100():
    return create_timm_model('ghostnet_100', 100, True)

def hornet_tiny():
    # 尝试不同的hornet模型名称
    model_names = [
        'hornet_tiny_7x7', 
        'hornet_tiny', 
        'hornet_tiny_224', 
        'hornet_nano'
    ]
    
    for name in model_names:
        model = create_timm_model(name, 100, True)
        if model is not None:
            return model
    
    # 如果都失败了，返回None或者使用一个替代模型
    print("警告: 所有HorNet模型都不可用，使用ConvNeXt作为替代")
    return create_timm_model('convnext_nano', 100, True)

def resnest50d():
    return create_timm_model('resnest50d', 100, True)

def mlp_mixer_b16():
    return create_timm_model('mixer_b16_224', 100, True)

# 模型映射字典
MODEL_REGISTRY = {
    'resnet_20': resnet20,
    'resnet_32': resnet32,
    'resnet_56': resnet56,
    'eca_resnet_20': eca_resnet20,
    'eca_resnet_32': eca_resnet32,
    'ghost_resnet_20': ghost_resnet20,
    'ghost_resnet_32': ghost_resnet32,
    'convnext_tiny': convnext_tiny,
    'convnext_tiny_timm': convnext_tiny_timm,
    'segnext_mscan_tiny': segnext_mscan_tiny,
    'coatnet_0': coatnet_0,
    'cspresnet50': cspresnet50,
    'ghostnet_100': ghostnet_100,
    'hornet_tiny': hornet_tiny,
    'resnest50d': resnest50d,
    'mlp_mixer_tiny': mlp_mixer_tiny,
    'mlp_mixer_b16': mlp_mixer_b16,
}

def get_model(model_name: str):
    """根据模型名称获取模型"""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]()
    else:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型: {model_name}. 可用模型: {available_models}")

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model_name: str):
    """获取模型信息"""
    model = get_model(model_name)
    params = count_parameters(model)
    return {
        'name': model_name,
        'parameters': params,
        'parameters_M': params / 1e6,
        'model': model
    } 