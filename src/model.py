import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Callable, Dict, Any

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

    def __init__(self, in_planes, planes, stride=1, **kwargs):
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
    def __init__(self, channels, k_size=3):
        super(ECALayer, self).__init__()
        if k_size % 2 == 0:
            k_size +=1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.k_size = k_size

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size=k_size)

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

class ECABasicBlock_Pos1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k_size=3):
        super(ECABasicBlock_Pos1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.eca1 = ECALayer(planes, k_size=k_size) # ECA after Conv1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.eca1(out) # Apply ECA after Conv1 and BN1, before ReLU1
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ECABasicBlock_Pos3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k_size=3):
        super(ECABasicBlock_Pos3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca_after_add = ECALayer(planes * self.expansion, k_size=k_size) # ECA after Add

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
        out = self.eca_after_add(out) # Apply ECA after Add, before final ReLU
        out = F.relu(out)
        return out

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        """
        args:
        - inp：输入通道数
        - oup：输出通道数
        - kernel_size：主卷积核大小
        - ratio：原始特征与廉价特征的比例
        - dw_size：深度可分离卷积核大小
        - stride：步长
        - relu：是否使用ReLU激活
        """
        super(GhostModule, self).__init__()
        self.oup = oup
      
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 主卷积部分
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 廉价操作部分（深度可分离卷积）
        self.cheap_operation = nn.Sequential()
        if new_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

    def forward(self, x):
        x1 = self.primary_conv(x)  
        # 廉价特征由主特征产生
        x2 = self.cheap_operation(x1) if self.cheap_operation else torch.zeros_like(x1)
        # 拼接特征并裁剪
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

# 用于构建 GhostNet 主干的残差块，包含两个 GhostModule 和残差连接
class GhostBasicBlock(nn.Module):
    expansion = 1  

    def __init__(self, in_planes, planes, stride=1, ratio=2):
        """
        args:
        - in_planes：输入特征图的通道数
        - planes：输出通道数（不乘以expansion）
        - stride：第一层GhostModule的步长，用于下采样
        - ratio：GhostModule中的通道扩展比
        """
        super(GhostBasicBlock, self).__init__()
        # 第一个GhostModule可进行下采样
        self.ghost1 = GhostModule(in_planes, planes, kernel_size=3, ratio=ratio, dw_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个GhostModule保持通道数与空间尺寸
        self.ghost2 = GhostModule(planes, planes, kernel_size=3, ratio=ratio, dw_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # 如果输入输出尺寸不一致，则构造shortcut进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.ghost1(x)))  # 第一次Ghost卷积并激活
        out = self.bn2(self.ghost2(out))        # 第二次Ghost卷积不激活
        out += self.shortcut(x)                 # 残差连接
        out = F.relu(out)                       # 最终激活
        return out

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        x = input_x + x
        return x
    

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class ImprovedBlock_ConvNeXt(nn.Module):
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
        
        # 使用 DropPath 进行正则化
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
        
        out = self.shortcut(input_x) + self.drop_path(out)
        out = F.relu(out)
        return out
    

class ImprovedBlock_ConvNeXt_NoDropPath(nn.Module):
    """消融实验: 无DropPath的版本"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        expand_ratio = 4
        expanded_planes = in_planes * expand_ratio

        self.conv1 = nn.Conv2d(in_planes, expanded_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_planes)
        
        self.dwconv = nn.Conv2d(
            expanded_planes, expanded_planes, kernel_size=7, 
            stride=stride, padding=3, groups=expanded_planes, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expanded_planes)
        
        self.conv2 = nn.Conv2d(expanded_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # 移除DropPath
        # self.drop_path = nn.Identity()

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
        
        out = self.shortcut(input_x) + out  # 直接相加，无DropPath
        out = F.relu(out)
        return out


class ImprovedBlock_ConvNeXt_StandardConv(nn.Module):
    """消融实验: 使用标准3x3卷积而非7x7深度卷积"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        expand_ratio = 4
        expanded_planes = in_planes * expand_ratio

        self.conv1 = nn.Conv2d(in_planes, expanded_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_planes)
        
        # 使用标准3x3卷积替代深度卷积
        self.conv_std = nn.Conv2d(
            expanded_planes, expanded_planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
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
        
        out = F.relu(self.bn1(self.conv1(x)))      # expand
        out = F.relu(self.bn2(self.conv_std(out))) # 标准卷积
        out = self.bn3(self.conv2(out))           # shrink
        
        out = self.shortcut(input_x) + self.drop_path(out)
        out = F.relu(out)
        return out


class ImprovedBlock_ConvNeXt_NoInvertedBottleneck(nn.Module):
    """消融实验：无倒置瓶颈，直接深度卷积"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0, **kwargs):
        super().__init__()
        
        # 直接使用深度卷积，无扩展
        self.dwconv = nn.Conv2d(
            in_planes, in_planes, kernel_size=7, 
            stride=stride, padding=3, groups=in_planes, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        self.conv_proj = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        input_x = x
        
        out = F.relu(self.bn1(self.dwconv(x)))    # 深度卷积
        out = self.bn2(self.conv_proj(out))       # 投影到输出维度
        
        out = self.shortcut(input_x) + self.drop_path(out)
        out = F.relu(out)
        return out


class MSCA(nn.Module):
    def __init__(self, dim, kernel_sizes=[7, 11], scale_factor=4):
        super().__init__()
        self.dim = dim
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.scales = nn.ModuleList()
        actual_kernel_sizes = [3, 5]
        
        self.scales.append(nn.Identity())
        for ks in actual_kernel_sizes:
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=(1, ks), padding=(0, ks//2), groups=dim),
                    nn.Conv2d(dim, dim, kernel_size=(ks, 1), padding=(ks//2, 0), groups=dim)
                )
            )
        
        self.conv_channel_mixer = nn.Conv2d(dim * (len(actual_kernel_sizes) +1) , dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        base_feat = self.conv0(x)
        
        scale_outputs = []
        for scale_conv in self.scales:
            scale_outputs.append(scale_conv(base_feat))
        
        concatenated_feats = torch.cat(scale_outputs, dim=1)
        
        attention_map = self.conv_channel_mixer(concatenated_feats)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class MSCABlock(nn.Module):
    def __init__(self, dim, drop_path=0., mlp_ratio=4.):
        super().__init__()
        self.msca = MSCA(dim)
        self.norm1 = nn.BatchNorm2d(dim)
        
        hidden_features = int(dim * mlp_ratio)
        self.conv1 = nn.Conv2d(dim, hidden_features, 1)
        self.conv2 = nn.Conv2d(hidden_features, dim, 1)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_x = self.msca(x_norm)
        x = attn_x + residual
        
        residual = x
        x_norm = self.norm1(x)
        x_mlp = self.conv1(x_norm)
        x_mlp = self.norm2(x_mlp)
        x_mlp = self.act(x_mlp)
        x_mlp = self.conv2(x_mlp)
        x = x_mlp + residual
        
        return x

class LSKBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim // 2, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map_base = self.conv0(x)
        spatial_attn = self.conv_spatial(attn_map_base)
        
        combined_attn = self.conv1(spatial_attn) 
        combined_attn = F.relu(combined_attn)
        combined_attn = self.conv2(combined_attn)
        attention_scores = self.sigmoid(combined_attn)
        
        return x * attention_scores

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

class ConvNeXtCustom(nn.Module):
    def __init__(self, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], num_classes=100, drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=1e-6))
            self.stages.append(nn.Sequential(*stage_blocks))
            if i < len(depths) - 1:
                self.stages.append(nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))
            cur += depths[i]

        self.norm_out = LayerNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        stage_idx = 0
        for i in range(len(self.depths)):
            x = self.stages[stage_idx](x)
            stage_idx +=1
            if i < len(self.depths) - 1:
                x = self.stages[stage_idx](x)
                stage_idx +=1
        return self.norm_out(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x
    

class ImprovedResNet_ConvNeXt(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=100, width_multiplier=1.0, 
                 drop_path_rate=0.05, in_channels=3):
        super().__init__()
        self.in_planes = int(16 * width_multiplier)
        self.cifar_stem = True  # 使用CIFAR风格的stem

        # CIFAR风格的stem
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # 计算每个block的drop path rate
        total_blocks = sum(num_blocks_list)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # 构建层
        planes_list = [16, 32, 64]
        strides = [1, 2, 2]
        
        current_planes = self.in_planes
        self.layers = nn.ModuleList()
        cur_idx = 0
        
        for i, num_blocks in enumerate(num_blocks_list):
            stage_planes = int(planes_list[i] * width_multiplier)
            stage_stride = strides[i]
            
            stage_dp_rates = dp_rates[cur_idx:cur_idx + num_blocks]
            self.layers.append(self._make_layer(block_type, stage_planes, num_blocks, 
                                              stride=stage_stride, dp_rates=stage_dp_rates))
            current_planes = stage_planes * block_type.expansion
            self.in_planes = current_planes
            cur_idx += num_blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(current_planes, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride, dp_rates):
        strides_list = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, (s, dp_rate) in enumerate(zip(strides_list, dp_rates)):
            layers.append(block_type(self.in_planes, planes, s, drop_path=dp_rate))
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



class MSCANEncoderCustom(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256], depths=[2, 2, 4, 2], num_classes=100, mlp_ratios=[4,4,4,4], drop_path_rate=0.0):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp_idx = 0
        
        for i in range(len(depths)):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(MSCABlock(dims[i], drop_path=dp_rates[cur_dp_idx + j], mlp_ratio=mlp_ratios[i]))
            
            self.stages.append(nn.Sequential(*stage_blocks))
            
            if i < len(depths) - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    nn.BatchNorm2d(dims[i+1])
                ))
            cur_dp_idx += depths[i]

        self.norm_out = nn.BatchNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        stage_idx = 0
        for i in range(len(self.depths)):
            x = self.stages[stage_idx](x)
            stage_idx += 1
            if i < len(self.depths) - 1:
                x = self.stages[stage_idx](x)
                stage_idx += 1
        return self.norm_out(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])
        x = self.head(x)
        return x

class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(out_features))
        self.weight_v = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        weight = F.normalize(self.weight_v, dim=1) * self.weight_g.unsqueeze(1)
        return F.linear(x, weight, self.bias)

# Swish自适应激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 门控网络，用于MoE选择专家
class GatingNetwork(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = WeightNormLinear(dim, num_experts)
        
    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

# 专家网络
class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation=Swish()):
        super().__init__()
        self.mlp = nn.Sequential(
            WeightNormLinear(in_dim, hidden_dim),
            activation,
            WeightNormLinear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

# 混合专家层
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, hidden_dim=None, activation=Swish(), capacity_factor=1.2):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, dim, activation) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(dim, num_experts)
        self.capacity_factor = capacity_factor
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        
        # 计算门控权重
        gates = self.gating(x_flat)  # [batch_size*seq_len, num_experts]
        
        # 选择top-k专家
        top_k = min(2, self.num_experts)
        top_values, top_indices = torch.topk(gates, top_k, dim=-1)
        
        # 专家路由
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (top_indices == i).any(dim=1).float().unsqueeze(1)
            if mask.sum() > 0:
                expert_input = x_flat * mask
                expert_outputs += mask * self.experts[i](expert_input)
        
        return expert_outputs.reshape(batch_size, seq_len, dim)

# 优化后的MixerBlock
class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout=0., 
                 use_moe=False, num_experts=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # 使用权重归一化和Swish激活的Token MLP
        self.token_mlp = nn.Sequential(
            WeightNormLinear(num_patches, token_mlp_dim),
            Swish(),
            nn.Dropout(dropout),
            WeightNormLinear(token_mlp_dim, num_patches),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        
        # 可选择使用MoE或标准MLP的Channel MLP
        if use_moe:
            self.channel_mlp = MoELayer(dim, num_experts=num_experts)
        else:
            self.channel_mlp = nn.Sequential(
                WeightNormLinear(dim, channel_mlp_dim),
                Swish(),
                nn.Dropout(dropout),
                WeightNormLinear(channel_mlp_dim, dim),
                nn.Dropout(dropout)
            )

    def forward(self, x):  # x shape: (B, num_patches, dim)
        # Token Mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)  # Shape: (B, dim, num_patches)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # Shape: (B, num_patches, dim)
        x = x + y

        # Channel Mixing
        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y
        return x

# 优化后的MLPMixerCustom
class MLPMixerCustom(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, token_mlp_dim, 
                 channel_mlp_dim, dropout=0., use_moe=False, num_experts=4, moe_layers=None):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # 指定哪些层使用MoE
        if moe_layers is None:
            moe_layers = [True] * depth if use_moe else [False] * depth
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, num_patches, token_mlp_dim, channel_mlp_dim, 
                      dropout=dropout, use_moe=moe_layers[i], num_experts=num_experts)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = WeightNormLinear(dim, num_classes)  # 输出层也使用权重归一化

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        skip_connection = x  # 保存初始特征作为跳跃连接
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = x + skip_connection  # 添加跳跃连接
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        x = self.head(x)
        return x

# --- New Model Implementations to replace TIMM dependencies ---

# --- GhostNet specific blocks (for GhostNet-100) ---
class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, kernel_size, stride, use_se=False, se_ratio=0.25):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        has_se = use_se

        self.ghost1 = GhostModule(in_chs, mid_chs, kernel_size=3, relu=True)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, kernel_size, stride=stride,
                                     padding=(kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModule(mid_chs, out_chs, kernel_size=1, relu=False)

        if stride == 1 and in_chs == out_chs:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, kernel_size=1, stride=stride,
                          padding=0, groups=in_chs, bias=False), # Corrected padding for 1x1 conv
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=torch.sigmoid):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = int(in_chs * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x_out = x * self.gate_fn(x_se)
        return x_out

class GhostNetCustom(nn.Module):
    def __init__(self, cfgs, num_classes=100, width=1.0, dropout=0.2, block=GhostBottleneck, se_module=SqueezeExcite):
        super(GhostNetCustom, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        output_channel = self._make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se, s in cfg:
                output_channel = self._make_divisible(c * width, 4)
                hidden_channel = self._make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se=se))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = self._make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(output_channel),
                                     nn.ReLU(inplace=True)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
        if self.dropout > 0.:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()


    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.classifier(x)
        return x

# --- CSPNet specific blocks (for CSPResNet50) ---
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, block_module, stride=1, use_eca=False, k_size=3):
        super().__init__()
        mid_channels = out_channels // 2
        
        # Path A (shortcut path, potentially strided)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Path B (main path with blocks, potentially strided before blocks)
        self.conv_shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(mid_channels)
        self.relu_shortcut = nn.ReLU(inplace=True) 
        
        self.blocks = self._make_layer(block_module, mid_channels, mid_channels, num_blocks, stride=1, use_eca=use_eca, k_size=k_size) # stride for internal blocks is 1
        
        self.conv_transition = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_transition = nn.BatchNorm2d(mid_channels)
        self.relu_transition = nn.ReLU(inplace=True)
        
        # Removed self.downsample_all initialization

        self.conv_final = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)


    def _make_layer(self, block_module, in_planes, planes, num_blocks, stride, use_eca=False, k_size=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        current_block_in_planes = in_planes # For BasicBlock, this will be mid_channels
        for s_val in strides:
            if use_eca:
                layers.append(block_module(current_block_in_planes, planes, stride=s_val, k_size=k_size))
            else:
                 layers.append(block_module(current_block_in_planes, planes, stride=s_val))
            current_block_in_planes = planes * block_module.expansion # For BasicBlock, planes*exp = mid_channels*1
        return nn.Sequential(*layers)

    def forward(self, x):
        x_shortcut_path_internal = self.relu1(self.bn1(self.conv1(x)))

        x_main_path_intermediate = self.relu_shortcut(self.bn_shortcut(self.conv_shortcut(x)))
        x_main_path_blocks = self.blocks(x_main_path_intermediate)
        x_main_path_final = self.relu_transition(self.bn_transition(self.conv_transition(x_main_path_blocks)))
        
        out = torch.cat((x_main_path_final, x_shortcut_path_internal), dim=1)
        out = self.relu_final(self.bn_final(self.conv_final(out)))
        return out

class CSPResNet(nn.Module):
    def __init__(self, block_module, csp_block_module, layers, num_classes=100, use_eca=False, k_size=3):
        super().__init__()
        self.in_planes = 64
        self.use_eca = use_eca
        self.k_size = k_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_csp_layer(block_module, csp_block_module, 64, layers[0], stride=1)
        self.layer2 = self._make_csp_layer(block_module, csp_block_module, 128, layers[1], stride=2)
        self.layer3 = self._make_csp_layer(block_module, csp_block_module, 256, layers[2], stride=2)
        self.layer4 = self._make_csp_layer(block_module, csp_block_module, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Calculate the correct in_features for the fc layer based on the output of the last CSP layer
        # The last CSP layer (layer4) uses planes=512. Its out_channels is 512 * expansion * 2.
        fc_in_features = 512 * block_module.expansion * 2
        self.fc = nn.Linear(fc_in_features, num_classes)

    def _make_csp_layer(self, block_module, csp_block_module, planes, num_blocks, stride):
        out_planes = planes * block_module.expansion * 2 # CSP doubles effective channels before final 1x1
        layer = csp_block_module(self.in_planes, out_planes, num_blocks, block_module, stride=stride, use_eca=self.use_eca, k_size=self.k_size)
        self.in_planes = out_planes
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- ResNeSt specific blocks (for ResNeSt50D) ---
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class SplitAttnConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, radix=2, reduction_factor=4, **kwargs):
        super(SplitAttnConv, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups # For ResNeSt, groups is cardinality
        self.channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * radix, kernel_size, stride, padding, dilation,
                              groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = nn.BatchNorm2d(out_channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, out_channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = RadixSoftmax(radix, self.cardinality)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class ResNeStBottleneckD(nn.Module): # 'D' variant for ResNeSt-D models
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=2, cardinality=1, bottleneck_width=64,
                 avd=True, avd_first=False, is_first=False):
        super(ResNeStBottleneckD, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first) # Apply AVD for strides or first block
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1 # Stride is handled by AvgPool

        self.conv2 = SplitAttnConv(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=1, groups=cardinality, bias=False,
            radix=radix)
        
        self.conv3 = nn.Conv2d(group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride # Store original stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        
        out = self.conv2(out)
        # SplitAttnConv has its own ReLU

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNeStCustom(nn.Module):
    def __init__(self, block, layers, num_classes=100, radix=2, cardinality=1, bottleneck_width=64,
                 deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False, dropout_rate=0.0): # Added dropout_rate
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNeStCustom, self).__init__()
        self.dropout_rate = dropout_rate # Store dropout_rate

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, avg_down=avg_down, is_first=True) # Added stride=1 and ensure is_first is correctly propagated
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down, is_first=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down, is_first=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down, is_first=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Correct in_features for ResNeSt fc layer
        fc_in_features = 512 * block.expansion # Should not be multiplied by 2 for ResNeSt
        
        fc_layers = []
        if self.dropout_rate > 0:
            fc_layers.append(nn.Dropout(self.dropout_rate))
        fc_layers.append(nn.Linear(fc_in_features, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def _make_layer(self, block, planes, blocks, stride, avg_down=True, is_first=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down: # For ResNeSt-D
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.radix,
                            self.cardinality, self.bottleneck_width, avd=self.avd,
                            avd_first=self.avd_first, is_first=is_first))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, # Explicitly set stride=1 for subsequent blocks
                                downsample=None, # Subsequent blocks in a stage typically don't downsample
                                radix=self.radix,
                                cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first, is_first=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- CoAtNet specific blocks (Simplified) ---
class MBConvBlock(nn.Module): # Simplified MBConv
    def __init__(self, inp, oup, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # pw
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # dw
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Squeeze-and-excitation
        if se_ratio > 0:
            squeeze_channels = max(1, int(inp * se_ratio))
            layers.append(SqueezeExcite(hidden_dim, reduced_base_chs=squeeze_channels, act_layer=nn.ReLU6)) # Use ReLU6 for SE like MobileNetV3

        # pw-linear
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Attention(nn.Module): # Simplified Self-Attention
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

class CoAtNetCustom(nn.Module): # Highly Simplified CoAtNet structure
    def __init__(self, num_classes=100,
                 s0_channels=64, s1_channels=96, s2_channels=192, s3_channels=384, s4_channels=768, # Example channel progression
                 s0_blocks=2, s1_blocks=2, s2_blocks=2, s3_blocks=2, s4_blocks=2, # Example block counts
                 mbconv_expand_ratio=4, transformer_heads=4, transformer_mlp_dim_ratio=4,
                 transformer_dropout=0.0): # Added transformer_dropout parameter
        super().__init__()
        self.num_classes = num_classes

        # Stem (S0)
        self.s0 = self._make_mbconv_stage(3, s0_channels, s0_blocks, stride=2, expand_ratio=mbconv_expand_ratio)

        # Convolutional stages (S1, S2)
        self.s1 = self._make_mbconv_stage(s0_channels, s1_channels, s1_blocks, stride=2, expand_ratio=mbconv_expand_ratio)
        self.s2 = self._make_mbconv_stage(s1_channels, s2_channels, s2_blocks, stride=2, expand_ratio=mbconv_expand_ratio)

        # Transformer stages (S3, S4)
        self.s3_transformer_blocks = self._make_transformer_stage(
            dim=s2_channels, num_blocks=s3_blocks, heads=transformer_heads,
            dim_head=s2_channels // transformer_heads,
            mlp_dim=s2_channels * transformer_mlp_dim_ratio,
            dropout_rate=transformer_dropout # Pass dropout to transformer stage
        )
        self.s3_pool = nn.MaxPool2d(kernel_size=2, stride=2) if s3_blocks > 0 else nn.Identity() # Downsample after S3 if transformers exist

        self.s4_transformer_blocks = self._make_transformer_stage(
            dim=s2_channels, # Assuming S3 output is pooled but channel dim remains from S2 for S4 input
            num_blocks=s4_blocks, heads=transformer_heads,
            dim_head=s2_channels // transformer_heads,
            mlp_dim=s2_channels * transformer_mlp_dim_ratio,
            dropout_rate=transformer_dropout # Pass dropout to transformer stage
        )
        
        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Determine the input dimension to the classifier
        # If S4 has blocks, its output dim is s2_channels (as per current simple structure)
        # If S4 has no blocks, but S3 does, its output dim is s2_channels
        # If S2 is the last feature stage, its output dim is s2_channels
        final_dim = s2_channels 
        self.fc = nn.Linear(final_dim, num_classes)

    def _make_mbconv_stage(self, in_c, out_c, num_blocks, stride, expand_ratio):
        layers = []
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            layers.append(MBConvBlock(in_c if i == 0 else out_c, out_c, current_stride, expand_ratio))
        return nn.Sequential(*layers)

    def _make_transformer_stage(self, dim, num_blocks, heads, dim_head, mlp_dim, dropout_rate):
        layers = []
        for _ in range(num_blocks):
            layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x) # Output spatial size H/8, W/8, channels s2_channels

        if len(self.s3_transformer_blocks) > 0:
            # Reshape for Transformer: (B, C, H, W) -> (B, H*W, C)
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_reshaped = self.s3_transformer_blocks(x_reshaped)
            # Reshape back: (B, H*W, C) -> (B, C, H, W)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w)
            x = self.s3_pool(x) # Downsample to H/16, W/16

        if len(self.s4_transformer_blocks) > 0:
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_reshaped = self.s4_transformer_blocks(x_reshaped)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- LSKNet Inspired Components ---
class LSKAttention(nn.Module):
    def __init__(self, channels, kernel_sizes: List[int], reduction_ratio=16):
        super().__init__()
        self.num_kernels = len(kernel_sizes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels * self.num_kernels, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = self.avg_pool(x)            # (b, c, 1, 1)
        attn = self.fc(attn)               # (b, c * num_kernels, 1, 1)
        attn = attn.view(b, self.num_kernels, c, 1, 1) # (b, num_kernels, c, 1, 1)
        attn = self.softmax(attn)          # Apply softmax over num_kernels dimension
        return attn # (b, num_kernels, c, 1, 1)

class MBConvBlock_enhanced(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, se_ratio=0.25, 
                 lsk_kernel_sizes: List[int] = [3, 5], lsk_reduction_ratio=16):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup
        self.lsk_kernel_sizes = lsk_kernel_sizes
        self.num_kernels = len(lsk_kernel_sizes)

        # Pointwise expansion
        if expand_ratio != 1:
            self.pw_expand = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.pw_expand = nn.Identity()

        # LSK-based Depthwise Convolution part
        self.lsk_dw_convs = nn.ModuleList()
        for k_size in lsk_kernel_sizes:
            self.lsk_dw_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, k_size, stride, k_size//2, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
            )
        self.lsk_attention = LSKAttention(hidden_dim, lsk_kernel_sizes, lsk_reduction_ratio)

        # Squeeze-and-excitation (optional)
        if se_ratio > 0:
            squeeze_channels = max(1, int(inp * se_ratio)) # SE based on input channels of the block
            self.se = SqueezeExcite(hidden_dim, reduced_base_chs=squeeze_channels, act_layer=nn.ReLU6)
        else:
            self.se = nn.Identity()

        # Pointwise linear projection
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        identity = x
        
        x = self.pw_expand(x)
        
        # LSK DW Conv
        dw_conv_outputs = []
        for conv_branch in self.lsk_dw_convs:
            dw_conv_outputs.append(conv_branch(x)) # (b, hidden_dim, h', w')
        
        # Get attention weights (b, num_kernels, hidden_dim, 1, 1)
        attn_weights = self.lsk_attention(x) 
        
        # Corrected weighted sum of DW conv outputs
        dw_conv_outputs_stacked = torch.stack(dw_conv_outputs, dim=1) # Shape: (b, num_kernels, hidden_dim, h', w')
        x_lsk = torch.sum(dw_conv_outputs_stacked * attn_weights, dim=1) # Shape: (b, hidden_dim, h', w')
        
        x = x_lsk # Output from LSK DW part

        x = self.se(x)
        x = self.pw_linear(x)

        if self.use_res_connect:
            return identity + x
        else:
            return x

# --- End LSKNet Inspired Components ---

# --- Enhanced CoAtNet with LSK-MBConv ---
class CoAtNetCustom_enhanced(nn.Module):
    def __init__(self, num_classes=100,
                 s0_channels=64, s1_channels=96, s2_channels=192, s3_channels=384, s4_channels=768,
                 s0_blocks=2, s1_blocks=2, s2_blocks=2, s3_blocks=2, s4_blocks=2,
                 mbconv_expand_ratio=4, transformer_heads=4, transformer_mlp_dim_ratio=4,
                 lsk_kernel_sizes: List[int] = [3, 5], lsk_reduction_ratio=16, se_ratio_in_mbconv=0.25,
                 transformer_dropout=0.0): # Added transformer_dropout
        super().__init__()
        self.num_classes = num_classes
        self.transformer_dropout = transformer_dropout # Store it

        # Stem (S0) - Enhanced MBConv
        self.s0 = self._make_mbconv_enhanced_stage(
            3, s0_channels, s0_blocks, stride=2, expand_ratio=mbconv_expand_ratio,
            lsk_kernel_sizes=lsk_kernel_sizes, lsk_reduction_ratio=lsk_reduction_ratio, se_ratio=se_ratio_in_mbconv
        )

        # Convolutional stages (S1, S2) - Enhanced MBConv
        self.s1 = self._make_mbconv_enhanced_stage(
            s0_channels, s1_channels, s1_blocks, stride=2, expand_ratio=mbconv_expand_ratio,
            lsk_kernel_sizes=lsk_kernel_sizes, lsk_reduction_ratio=lsk_reduction_ratio, se_ratio=se_ratio_in_mbconv
        )
        self.s2 = self._make_mbconv_enhanced_stage(
            s1_channels, s2_channels, s2_blocks, stride=2, expand_ratio=mbconv_expand_ratio,
            lsk_kernel_sizes=lsk_kernel_sizes, lsk_reduction_ratio=lsk_reduction_ratio, se_ratio=se_ratio_in_mbconv
        )

        # Transformer stages (S3, S4)
        self.s3_transformer_blocks = self._make_transformer_stage(
            dim=s2_channels, num_blocks=s3_blocks, heads=transformer_heads,
            dim_head=s2_channels // transformer_heads,
            mlp_dim=s2_channels * transformer_mlp_dim_ratio,
            dropout_rate=self.transformer_dropout # Use stored dropout
        )
        self.s3_pool = nn.MaxPool2d(kernel_size=2, stride=2) if s3_blocks > 0 else nn.Identity()

        self.s4_transformer_blocks = self._make_transformer_stage(
            dim=s2_channels,
            num_blocks=s4_blocks, heads=transformer_heads,
            dim_head=s2_channels // transformer_heads,
            mlp_dim=s2_channels * transformer_mlp_dim_ratio,
            dropout_rate=self.transformer_dropout # Use stored dropout
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        final_dim = s2_channels
        self.fc = nn.Linear(final_dim, num_classes)


    def _make_mbconv_enhanced_stage(self, in_c, out_c, num_blocks, stride, expand_ratio, lsk_kernel_sizes, lsk_reduction_ratio, se_ratio):
        layers = []
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            layers.append(MBConvBlock_enhanced(
                in_c if i == 0 else out_c, out_c, current_stride, expand_ratio,
                lsk_kernel_sizes=lsk_kernel_sizes, lsk_reduction_ratio=lsk_reduction_ratio, se_ratio=se_ratio
            ))
        return nn.Sequential(*layers)

    def _make_transformer_stage(self, dim, num_blocks, heads, dim_head, mlp_dim, dropout_rate):
        layers = []
        for _ in range(num_blocks):
            layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)

        if len(self.s3_transformer_blocks) > 0:
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_reshaped = self.s3_transformer_blocks(x_reshaped)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w)
            x = self.s3_pool(x)

        if len(self.s4_transformer_blocks) > 0:
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_reshaped = self.s4_transformer_blocks(x_reshaped)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- End Enhanced CoAtNet ---


# --- Model Registry and Getter ---
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str) -> Callable[..., Callable[..., nn.Module]]:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = builder
        return builder
    return decorator

@register_model("resnet_20")
def resnet20_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)

@register_model("resnet_32")
def resnet32_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, **kwargs)

@register_model("resnet_56")
def resnet56_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, **kwargs)

@register_model("eca_resnet_20")
def eca_resnet20_builder(num_classes=100, k_size=3, **kwargs):
    return ResNet(ECABasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs={'k_size': k_size}, **kwargs)

@register_model("eca_resnet_32")
def eca_resnet32_builder(num_classes=100, k_size=5, **kwargs): # k_size default 5 for eca_r32
    return ResNet(ECABasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs={'k_size': k_size}, **kwargs)

@register_model("ghost_resnet_20")
def ghost_resnet20_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs={'ratio': ratio}, **kwargs)

@register_model("ghost_resnet_32")
def ghost_resnet32_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs={'ratio': ratio}, **kwargs)

@register_model("ghostnet_100") # Corresponds to GhostNet 1.0x
def ghostnet_100_builder(num_classes=100, width=1.0, dropout=0.2, **kwargs):
    # GhostNetV1 configurations: (kernel_size, hidden_expansion_size, out_channels, use_se, stride)
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 1, 2]], # use_se=1 for SqueezeExcite
        [[5, 120,  40, 1, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 1, 1],
         [3, 672, 112, 1, 1]
        ],
        # stage5
        [[5, 672, 160, 1, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 1, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 1, 1]
        ]
    ]
    return GhostNetCustom(cfgs, num_classes=num_classes, width=width, dropout=dropout, **kwargs)

@register_model("convnext_tiny")
def convnext_tiny_custom_builder(num_classes=100, **kwargs):
    # Depths and Dims for ConvNeXt-Tiny like from official implementation
    return ConvNeXtCustom(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)

@register_model("improved_resnet20_convnext")
def improved_resnet20_convnext_builder(num_classes=100, width_multiplier=1.0, drop_path_rate=0.05, **kwargs):
    """ResNet20 + Depthwise Conv + Inverted Bottleneck"""
    return ImprovedResNet_ConvNeXt(ImprovedBlock_ConvNeXt, [3, 3, 3], num_classes=num_classes, 
                         width_multiplier=width_multiplier, drop_path_rate=drop_path_rate, **kwargs)

@register_model("improved_resnet20_convnext_no_droppath")
def improved_resnet20_convnext_no_droppath_builder(num_classes=100, width_multiplier=1.0, drop_path_rate=0.0, **kwargs):
    """消融实验: 无DropPath的版本"""
    return ImprovedResNet_ConvNeXt(ImprovedBlock_ConvNeXt_NoDropPath, [3, 3, 3], num_classes=num_classes, 
                         width_multiplier=width_multiplier, drop_path_rate=drop_path_rate, **kwargs)

@register_model("improved_resnet20_convnext_std_conv")
def improved_resnet20_convnext_std_conv_builder(num_classes=100, width_multiplier=1.0, drop_path_rate=0.05, **kwargs):
    """消融实验: 使用标准3x3卷积"""
    return ImprovedResNet_ConvNeXt(ImprovedBlock_ConvNeXt_StandardConv, [3, 3, 3], num_classes=num_classes, 
                         width_multiplier=width_multiplier, drop_path_rate=drop_path_rate, **kwargs)

@register_model("improved_resnet20_convnext_no_inverted")
def improved_resnet20_convnext_no_inverted_builder(num_classes=100, width_multiplier=1.0, drop_path_rate=0.05, **kwargs):
    """消融实验: 无倒置瓶颈"""
    return ImprovedResNet_ConvNeXt(ImprovedBlock_ConvNeXt_NoInvertedBottleneck, [3, 3, 3], num_classes=num_classes, 
                         width_multiplier=width_multiplier, drop_path_rate=drop_path_rate, **kwargs)
                         
@register_model("segnext_mscan_tiny") # This seems to be MSCANEncoderCustom from before
def segnext_mscan_tiny_custom_builder(num_classes=100, **kwargs):
    # Dims and depths for a "Tiny" variant, e.g. like SegNeXt-T
    return MSCANEncoderCustom(dims=[32, 64, 160, 256], depths=[3, 3, 5, 2], num_classes=num_classes, **kwargs)

@register_model("mlp_mixer_tiny")
def mlp_mixer_tiny_custom_builder(num_classes=100, **kwargs):
    # Default "Tiny" parameters for CIFAR-100 like
    return MLPMixerCustom(
    image_size=32,
    patch_size=4,
    dim=128,
    depth=8,
    num_classes=100,
    token_mlp_dim=256,
    channel_mlp_dim=512,
    use_moe=True,
    num_experts=4,
    moe_layers=[i % 2 == 0 for i in range(8)])  # 仅在偶数层使用MoE

@register_model("mlp_mixer_b16")
def mlp_mixer_b16_builder(num_classes=100, **kwargs):
    # "Base" parameters for MLP-Mixer, patch size 16. Adjusted for CIFAR.
    # Original B/16 is dim=768, depth=12. For CIFAR, might be too large.
    # Using a scaled down version or parameters inspired by timm's cifar adaptations if any.
    # Let's use dim=512, depth=8, patch_size=4 (as 32/16=2x2 patches is too few)
    return MLPMixerCustom(image_size=32, patch_size=4, dim=512, depth=8, 
                          token_mlp_dim=1024, channel_mlp_dim=2048, num_classes=num_classes, **kwargs)


@register_model("cspresnet50")
def cspresnet50_builder(num_classes=100, **kwargs):
    # ResNet50 layers: [3, 4, 6, 3] for Bottleneck. BasicBlock used here needs adjustment.
    # Assuming BasicBlock based CSPNet for now, as Bottleneck version is more complex.
    # Or, implement Bottleneck and CSPBottleneck. For simplicity, using BasicBlock based structure.
    # To be a true CSPResNet50, it would need Bottleneck blocks.
    # This will be a CSPNet with ResNet50-like depth using BasicBlocks.
    return CSPResNet(BasicBlock, CSPBlock, [3,4,6,3], num_classes=num_classes, **kwargs)

@register_model("resnest50d")
def resnest50d_builder(num_classes=100, **kwargs):
    # Extract dropout_rate from kwargs, defaulting to 0.0 if not present
    dropout_val = kwargs.pop('dropout_rate', 0.0) 
    return ResNeStCustom(ResNeStBottleneckD, [3, 4, 6, 3], num_classes=num_classes,
                         radix=2, cardinality=1, bottleneck_width=64,
                         deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False, 
                         dropout_rate=dropout_val, # Pass dropout_rate to constructor
                         **kwargs)

@register_model("coatnet_0") # Builder for the non-enhanced CoAtNetCustom
def coatnet_0_builder(num_classes=100, **kwargs):
    # Default CoAtNet-0 parameters
    transformer_dropout_val = kwargs.pop('transformer_dropout_rate', 0.0) # Extract and remove to prevent passing to deeper unused places
    
    # Simplified structure, adjust sX_channels and sX_blocks as per CoAtNet-0 needs
    # These are example values, original CoAtNet-0 has specific block counts and channel sizes for each stage
    return CoAtNetCustom(num_classes=num_classes,
                         s0_blocks=kwargs.get('s0_blocks', 2), 
                         s1_blocks=kwargs.get('s1_blocks', 2), 
                         s2_blocks=kwargs.get('s2_blocks', 3),  # More MBConv
                         s3_blocks=kwargs.get('s3_blocks', 5),  # More Transformer
                         s4_blocks=kwargs.get('s4_blocks', 2),  # More Transformer
                         s0_channels=kwargs.get('s0_channels', 64), 
                         s1_channels=kwargs.get('s1_channels', 128), 
                         s2_channels=kwargs.get('s2_channels', 256), # Input to S3/S4 Transformer
                         # s3_channels and s4_channels are not directly used for Transformer dim in this simplified model
                         transformer_heads=kwargs.get('transformer_heads', 8),
                         transformer_dropout=transformer_dropout_val, # Pass the extracted dropout
                         **kwargs)

@register_model("coatnet_0_custom_enhanced")
def coatnet_0_enhanced_builder(num_classes=100, **kwargs):
    transformer_dropout_val = kwargs.pop('transformer_dropout_rate', 0.0)
    lsk_kernel_sizes = kwargs.get('lsk_kernel_sizes', [3,5,7]) # Default LSK if not provided
    lsk_reduction_ratio = kwargs.get('lsk_reduction_ratio', 8)
    se_ratio_in_mbconv = kwargs.get('se_ratio_in_mbconv', 0.25)

    return CoAtNetCustom_enhanced(
        num_classes=num_classes,
        s0_blocks=kwargs.get('s0_blocks', 2), 
        s1_blocks=kwargs.get('s1_blocks', 2), 
        s2_blocks=kwargs.get('s2_blocks', 3),
        s3_blocks=kwargs.get('s3_blocks', 5),
        s4_blocks=kwargs.get('s4_blocks', 2),
        s0_channels=kwargs.get('s0_channels', 64), 
        s1_channels=kwargs.get('s1_channels', 128), 
        s2_channels=kwargs.get('s2_channels', 256),
        transformer_heads=kwargs.get('transformer_heads', 8),
        transformer_dropout=transformer_dropout_val,
        lsk_kernel_sizes=lsk_kernel_sizes,
        lsk_reduction_ratio=lsk_reduction_ratio,
        se_ratio_in_mbconv=se_ratio_in_mbconv,
        **kwargs
    )

def get_model(model_name: str, num_classes: int = 100, **kwargs: Any) -> nn.Module:
    """
    Retrieves a model instance from the registry.
    Args:
        model_name (str): Name of the model (must be registered).
        num_classes (int): Number of output classes.
        kwargs (Any): Additional arguments for the model builder.
    Returns:
        nn.Module: Instantiated PyTorch model.
    Raises:
        ValueError: If the model_name is not found in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        # Check for legacy timm model names and map or raise error
        if model_name == "convnext_tiny_timm":
            print(f"Warning: Model '{model_name}' is deprecated. Using 'convnext_tiny' (custom implementation).")
            model_name = "convnext_tiny"
        elif model_name == "ghostnet_100_timm": # Assuming timm's ghostnet_100 is similar to our ghostnet_100
             print(f"Warning: Model '{model_name}' is deprecated. Using 'ghostnet_100' (custom implementation).")
             model_name = "ghostnet_100"
        elif model_name == "mlp_mixer_b16_timm":
            print(f"Warning: Model '{model_name}' is deprecated. Using 'mlp_mixer_b16' (custom implementation).")
            model_name = "mlp_mixer_b16"
        elif model_name in ["coatnet_0", "cspresnet50_timm", "hornet_tiny_timm", "resnest50d_timm"]:
             # For models that had distinct _timm versions and now have _custom
             # prefer _custom if available.
             custom_name = model_name.replace("_timm", "_custom") if "_timm" in model_name else model_name + "_custom"
             if custom_name in MODEL_REGISTRY:
                 print(f"Warning: Model '{model_name}' (timm version) is deprecated. Using '{custom_name}'.")
                 model_name = custom_name
             elif model_name == "cspresnet50_timm": # Map to the new cspresnet50
                  print(f"Warning: Model '{model_name}' (timm version) is deprecated. Using 'cspresnet50'.")
                  model_name = "cspresnet50"
             elif model_name == "resnest50d_timm":
                  print(f"Warning: Model '{model_name}' (timm version) is deprecated. Using 'resnest50d'.")
                  model_name = "resnest50d"
             else:
                raise ValueError(f"Model '{model_name}' not found in registry and no direct custom replacement available. Native PyTorch implementation needed.")
        else:
             raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY.")

    builder = MODEL_REGISTRY[model_name]
    print(f"Building model: {model_name} with kwargs: {kwargs}")
    return builder(num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> float:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # Return in Millions

def get_model_info(model: nn.Module, model_name: str) -> Dict[str, Any]:
    """
    Gets basic information about the model.
    Args:
        model (nn.Module): The model instance.
        model_name (str): The name of the model.
    Returns:
        dict: Dictionary with model information (e.g., params_M).
    """
    params_m = count_parameters(model)
    print(f'{model_name} - Parameters: {params_m:.2f}M')
    return {
        'model_name': model_name,
        'params_M': params_m,
    }

if __name__ == '__main__':
    # Test model registration and instantiation
    test_model_names = [
        "resnet_56", "eca_resnet_20", "ghost_resnet_32", 
        "convnext_tiny", "segnext_mscan_tiny", "mlp_mixer_tiny",
        "ghostnet_100", "cspresnet50", "resnest50d",
        "mlp_mixer_b16", "coatnet_0", "coatnet_0_custom_enhanced" # Added coatnet_0
    ]
    
    for name in test_model_names:
        print(f"--- Testing {name} ---")
        try:
            # Test with default num_classes=100
            model_instance = get_model(name, num_classes=100)
            info = get_model_info(model_instance, name)
            # Test a forward pass with dummy data
            dummy_input = torch.randn(2, 3, 32, 32) # CIFAR-100 like
            if name == "mlp_mixer_tiny" or name == "mlp_mixer_b16": # MLP Mixers don't take 4D HxW input directly sometimes
                pass # Their custom forward handles it / or it's for 224x224. For CIFAR, it's ok.
            
            output = model_instance(dummy_input)
            print(f"Output shape for {name}: {output.shape}")
            assert output.shape == (2, 100)
            print(f"{name} test passed.\n")

        except Exception as e:
            print(f"Error testing model {name}: {e}\n")

    # Test a model that might need specific kwargs
    print("--- Testing eca_resnet_20 with k_size ---")
    try:
        model_instance = get_model("eca_resnet_20", num_classes=10, k_size=5) # Override k_size
        info = get_model_info(model_instance, "eca_resnet_20_custom_ksize")
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model_instance(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 10)
        print("eca_resnet_20 with k_size test passed.\n")
    except Exception as e:
        print(f"Error testing eca_resnet_20 with k_size: {e}\n") 

    # Add a specific test for coatnet_0_custom_enhanced with some LSK kargs
    print("--- Testing coatnet_0_custom_enhanced with specific LSK kwargs ---")
    try:
        model_instance_enhanced = get_model(
            "coatnet_0_custom_enhanced", 
            num_classes=10, 
            lsk_kernel_sizes=[3, 5], 
            lsk_reduction_ratio=4,
            s0_blocks=1, s1_blocks=1, s2_blocks=1, s3_blocks=1, s4_blocks=1 # Smaller model for quick test
        )
        info = get_model_info(model_instance_enhanced, "coatnet_0_custom_enhanced_specific_lsk")
        dummy_input = torch.randn(2, 3, 32, 32) # CIFAR-100 like
        output = model_instance_enhanced(dummy_input)
        print(f"Output shape for coatnet_0_custom_enhanced_specific_lsk: {output.shape}")
        assert output.shape == (2, 10)
        print("coatnet_0_custom_enhanced with specific LSK kwargs test passed.\n")
    except Exception as e:
        print(f"Error testing coatnet_0_custom_enhanced with specific LSK kwargs: {e}\n")


    # 测试改进的 improved_resnet20_convnext 模型
    improved_test_models = [
        "improved_resnet20_convnext",
        "improved_resnet20_convnext_no_droppath",  # 无DropPath
        "improved_resnet20_convnext_std_conv",     # 标准卷积
        "improved_resnet20_convnext_no_inverted",  # 无倒置瓶颈
    ]
    
    print("\n--- Testing Improved ResNet v2 Models ---")
    for name in improved_test_models:
        print(f"--- Testing {name} ---")
        try:
            model_instance = get_model(name, num_classes=100)
            info = get_model_info(model_instance, name)
            dummy_input = torch.randn(2, 3, 32, 32)
            output = model_instance(dummy_input)
            print(f"Output shape for {name}: {output.shape}")
            assert output.shape == (2, 100)
            print(f"{name} test passed.\n")
        except Exception as e:
            print(f"Error testing model {name}: {e}\n")

# --- CoAtNet-CIFAROpt Implementation ---
class ECAMBConvBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, k_size=3):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        # Pointwise expansion
        if expand_ratio != 1:
            self.pw_expand = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.pw_expand = nn.Identity()

        # Depthwise convolution
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # ECA Module (replacing SE)
        self.eca = ECALayer(hidden_dim, k_size=k_size)

        # Pointwise linear projection
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        identity = x
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.eca(x)
        x = self.pw_linear(x)
        
        if self.use_res_connect:
            return identity + x
        else:
            return x

class ECATransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., eca_k_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN with ECA integration
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Add ECA to channel dimension after FFN
        self.channel_eca = ECALayer(dim, k_size=eca_k_size)

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x))
        
        # FFN with ECA
        ffn_out = x + self.ffn(self.norm2(x))
        
        # Apply ECA to channel dimension
        # Need to reshape for ECA: (B, N, C) -> (B, C, 1, N) -> (B, C, 1, N) -> (B, N, C)
        b, n, c = ffn_out.shape
        ffn_out_2d = ffn_out.transpose(1, 2).unsqueeze(2)  # (B, C, 1, N)
        eca_out = self.channel_eca(ffn_out_2d)  # (B, C, 1, N)
        result = eca_out.squeeze(2).transpose(1, 2)  # (B, N, C)
        
        return result

class CoAtNetCIFAROpt(nn.Module):
    def __init__(self, num_classes=100,
                 s0_channels=64, s1_channels=96, s2_channels=192, s3_channels=384, s4_channels=512,
                 s0_blocks=2, s1_blocks=2, s2_blocks=6, s3_blocks=10, s4_blocks=2,
                 mbconv_expand_ratio=4, transformer_heads=8, transformer_mlp_dim_ratio=4,
                 eca_k_size=3, stem_kernel_size=3, dropout=0.0, transformer_dropout=0.0): # Added transformer_dropout
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout # General dropout for final layer if needed
        self.transformer_dropout = transformer_dropout # Specific for transformer blocks

        # Stem (S0)
        padding = (stem_kernel_size - 1) // 2
        self.s0_stem = nn.Sequential(
            nn.Conv2d(3, s0_channels, kernel_size=stem_kernel_size, stride=1, padding=padding, bias=False), # stride 1 for CIFAR
            nn.BatchNorm2d(s0_channels),
            nn.ReLU(inplace=True)
        )
        # MBConv blocks for S0 after stem
        self.s0_blocks_seq = self._make_eca_mbconv_stage(
            s0_channels, s0_channels, s0_blocks, stride=1, # Stride 1 as stem already handled initial feature map
            expand_ratio=mbconv_expand_ratio, eca_k_size=eca_k_size
        )
        
        # Convolutional stages (S1, S2) with ECA-MBConv
        self.s1 = self._make_eca_mbconv_stage(
            s0_channels, s1_channels, s1_blocks, stride=2, 
            expand_ratio=mbconv_expand_ratio, eca_k_size=eca_k_size
        ) # Out: C=s1_channels, H/2, W/2 (if initial was H,W) -> For CIFAR (32->16)
        
        self.s2 = self._make_eca_mbconv_stage(
            s1_channels, s2_channels, s2_blocks, stride=2, 
            expand_ratio=mbconv_expand_ratio, eca_k_size=eca_k_size
        ) # Out: C=s2_channels, H/4, W/4 (16->8)

        # Transformer stages (S3, S4) with ECA-Transformer
        self.s3_transformer_blocks = self._make_eca_transformer_stage(
            dim=s2_channels, num_blocks=s3_blocks, heads=transformer_heads,
            dim_head=s2_channels // transformer_heads,
            mlp_dim=s2_channels * transformer_mlp_dim_ratio,
            dropout=self.transformer_dropout, # Use the passed transformer_dropout
            eca_k_size=eca_k_size
        ) # Operates on H/4, W/4 features
        self.s3_pool = nn.MaxPool2d(kernel_size=2, stride=2) if s3_blocks > 0 and s4_blocks > 0 else nn.Identity() 
        # Pool if S3 exists AND S4 will follow. If S3 is last transformer stage, no pool before classifier.

        current_dim_for_s4 = s2_channels # Dim after S2, or after S3 pooling if S3 happened
        
        self.s4_transformer_blocks = self._make_eca_transformer_stage(
            dim=current_dim_for_s4, num_blocks=s4_blocks, heads=transformer_heads,
            dim_head=current_dim_for_s4 // transformer_heads,
            mlp_dim=current_dim_for_s4 * transformer_mlp_dim_ratio,
            dropout=self.transformer_dropout, # Use the passed transformer_dropout
            eca_k_size=eca_k_size
        ) # Operates on H/8, W/8 features if s3_pool happened

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        final_classifier_dim = s2_channels if s4_blocks == 0 else current_dim_for_s4 # Or s3_channels if s3 has diff dim
        if s4_blocks > 0:
            final_classifier_dim = current_dim_for_s4
        elif s3_blocks > 0: # S3 is last, S4 is not
            final_classifier_dim = s2_channels # S3 operates on s2_channels dim
        else: # S2 is last feature stage
            final_classifier_dim = s2_channels

        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate if self.dropout_rate > 0 else 0.0), # General dropout before FC
            nn.Linear(final_classifier_dim, num_classes)
        )

    def _make_eca_mbconv_stage(self, in_c, out_c, num_blocks, stride, expand_ratio, eca_k_size):
        layers = []
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            layers.append(ECAMBConvBlock(in_c if i == 0 else out_c, out_c, current_stride, expand_ratio, k_size=eca_k_size))
        return nn.Sequential(*layers)

    def _make_eca_transformer_stage(self, dim, num_blocks, heads, dim_head, mlp_dim, dropout, eca_k_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(ECATransformerBlock(dim, heads, dim_head, mlp_dim, dropout=dropout, eca_k_size=eca_k_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: 32x32
        x = self.s0_stem(x)      # Out: C=s0_channels, 32x32
        x = self.s0_blocks_seq(x) # Out: C=s0_channels, 32x32
        
        x = self.s1(x)           # Out: C=s1_channels, 16x16
        x = self.s2(x)           # Out: C=s2_channels, 8x8

        if len(self.s3_transformer_blocks) > 0:
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2) # (B, H*W, C)
            x_reshaped = self.s3_transformer_blocks(x_reshaped)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w) # (B, C, H, W)
            x = self.s3_pool(x) # Potentially H/2, W/2 => 4x4 if S4 exists

        if len(self.s4_transformer_blocks) > 0:
            b, c, h, w = x.shape
            x_reshaped = x.flatten(2).transpose(1, 2)
            x_reshaped = self.s4_transformer_blocks(x_reshaped)
            x = x_reshaped.transpose(1, 2).reshape(b, c, h, w)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@register_model("coatnet_cifar_opt")
def coatnet_cifar_opt_builder(num_classes=100, **kwargs):
    transformer_dropout_val = kwargs.pop('transformer_dropout_rate', 0.0)
    dropout_val = kwargs.pop('dropout', 0.1) # General dropout before FC
    
    return CoAtNetCIFAROpt(
        num_classes=num_classes,
        s0_channels=kwargs.get('s0_channels', 64), 
        s1_channels=kwargs.get('s1_channels', 96), 
        s2_channels=kwargs.get('s2_channels', 192),
        s3_channels=kwargs.get('s3_channels', 384), # Not directly used for dim in this simplified setup for S3/S4
        s4_channels=kwargs.get('s4_channels', 512), # Not directly used for dim
        s0_blocks=kwargs.get('s0_blocks', 2), 
        s1_blocks=kwargs.get('s1_blocks', 2), 
        s2_blocks=kwargs.get('s2_blocks', 3), 
        s3_blocks=kwargs.get('s3_blocks', 5), 
        s4_blocks=kwargs.get('s4_blocks', 2),
        mbconv_expand_ratio=kwargs.get('mbconv_expand_ratio', 4),
        transformer_heads=kwargs.get('transformer_heads', 4), # Fewer heads for smaller model
        transformer_mlp_dim_ratio=kwargs.get('transformer_mlp_dim_ratio', 2), # Smaller MLP ratio
        eca_k_size=kwargs.get('eca_k_size', 3),
        stem_kernel_size=kwargs.get('stem_kernel_size', 3),
        dropout=dropout_val, # General dropout
        transformer_dropout=transformer_dropout_val, # Transformer specific dropout
        **kwargs
    )

@register_model("coatnet_cifar_opt_large_stem")
def coatnet_cifar_opt_large_stem_builder(num_classes=100, **kwargs):
    transformer_dropout_val = kwargs.pop('transformer_dropout_rate', 0.0)
    dropout_val = kwargs.pop('dropout', 0.1)

    return CoAtNetCIFAROpt(
        num_classes=num_classes,
        s0_channels=kwargs.get('s0_channels', 64), 
        s1_channels=kwargs.get('s1_channels', 96), 
        s2_channels=kwargs.get('s2_channels', 192),
        s3_channels=kwargs.get('s3_channels', 384),
        s4_channels=kwargs.get('s4_channels', 512),
        s0_blocks=kwargs.get('s0_blocks', 2), 
        s1_blocks=kwargs.get('s1_blocks', 2), 
        s2_blocks=kwargs.get('s2_blocks', 3), 
        s3_blocks=kwargs.get('s3_blocks', 5), 
        s4_blocks=kwargs.get('s4_blocks', 2),
        mbconv_expand_ratio=kwargs.get('mbconv_expand_ratio', 4),
        transformer_heads=kwargs.get('transformer_heads', 4),
        transformer_mlp_dim_ratio=kwargs.get('transformer_mlp_dim_ratio', 2),
        eca_k_size=kwargs.get('eca_k_size', 3),
        stem_kernel_size=kwargs.get('stem_kernel_size', 5), # Larger stem
        dropout=dropout_val,
        transformer_dropout=transformer_dropout_val,
        **kwargs
    )


#### ECA-Net 20 Comparison model ####
class AdaptiveECALayer(nn.Module):
    """自适应ECA层，根据通道数自动计算核大小"""
    def __init__(self, channels, gamma=2, b=1):
        super(AdaptiveECALayer, self).__init__()
        # 根据通道数自适应计算核大小
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

class ECAAdaptiveBasicBlock(nn.Module):
    """ECA + 自适应核大小的基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(ECAAdaptiveBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = AdaptiveECALayer(planes)

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

class ECAFixedBasicBlock(nn.Module):
    """ECA + 固定k=3的基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(ECAFixedBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size=3)  # 固定k=3

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

class NoECABasicBlock(nn.Module):
    """无ECA的基础块（与原始BasicBlock相同）"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(NoECABasicBlock, self).__init__()
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

class ECAFixedK5BasicBlock(nn.Module):
    """ECA + 固定k=5的基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(ECAFixedK5BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size=5)  # 固定k=5

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

class ECAFixedK7BasicBlock(nn.Module):
    """ECA + 固定k=7的基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(ECAFixedK7BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size=7)  # 固定k=7

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

class ECAFixedK9BasicBlock(nn.Module):
    """ECA + 固定k=9的基础块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(ECAFixedK9BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size=9)  # 固定k=9

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

@register_model("ecanet20_fixed_k5")
def ecanet20_fixed_k5_builder(num_classes=100, **kwargs):
    """ECA + 固定k=5的ECANet20"""
    # 过滤掉 k_size 参数，因为 ECAFixedK5BasicBlock 内部已经硬编码了 k_size=5
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(ECAFixedK5BasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

@register_model("ecanet20_fixed_k7")
def ecanet20_fixed_k7_builder(num_classes=100, **kwargs):
    """ECA + 固定k=7的ECANet20"""
    # 过滤掉 k_size 参数，因为 ECAFixedK7BasicBlock 内部已经硬编码了 k_size=7
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(ECAFixedK7BasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

@register_model("ecanet20_fixed_k9")
def ecanet20_fixed_k9_builder(num_classes=100, **kwargs):
    """ECA + 固定k=9的ECANet20"""
    # 过滤掉 k_size 参数，因为 ECAFixedK9BasicBlock 内部已经硬编码了 k_size=9
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(ECAFixedK9BasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

# 同样修正其他已有的构建函数
@register_model("ecanet20_adaptive")
def ecanet20_adaptive_builder(num_classes=100, **kwargs):
    """ECA + 自适应核大小的ECANet20"""
    # 过滤掉 k_size 参数，因为 ECAAdaptiveBasicBlock 内部使用自适应计算
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(ECAAdaptiveBasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

@register_model("ecanet20_fixed_k3")
def ecanet20_fixed_k3_builder(num_classes=100, **kwargs):
    """ECA + 固定k=3的ECANet20"""
    # 过滤掉 k_size 参数，因为 ECAFixedBasicBlock 内部已经硬编码了 k_size=3
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(ECAFixedBasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

@register_model("resnet20_no_eca")
def resnet20_no_eca_builder(num_classes=100, **kwargs):
    """无ECA的ResNet20（参考模型）"""
    # 过滤掉 k_size 参数，因为 NoECABasicBlock 不使用 ECA
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'k_size'}
    return ResNet(NoECABasicBlock, [3, 3, 3], num_classes=num_classes, **filtered_kwargs)

@register_model("eca_resnet20_pos1")
def eca_resnet20_pos1_builder(num_classes=100, k_size=3, **kwargs):
    return ResNet(ECABasicBlock_Pos1, [3, 3, 3], num_classes=num_classes, block_kwargs={'k_size': k_size}, **kwargs)

@register_model("eca_resnet20_pos3")
def eca_resnet20_pos3_builder(num_classes=100, k_size=3, **kwargs):
    return ResNet(ECABasicBlock_Pos3, [3, 3, 3], num_classes=num_classes, block_kwargs={'k_size': k_size}, **kwargs)
#### ECA-Net 20 Comparison model ####