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
        if hasattr(self.cheap_operation, 'conv'):
             x2 = self.cheap_operation(x1)
             out = torch.cat([x1, x2], dim=1)
        else:
             out = x1
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

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_patches),
            nn.Dropout(dropout)
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_token_mixed = self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_token_mixed
        x_channel_mixed = self.channel_mix(x)
        x = x + x_channel_mixed
        return x

class MLPMixerCustom(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, token_mlp_dim, channel_mlp_dim, dropout=0.):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, num_patches, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, dim]
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.norm(x)
        x = x.mean(dim=1) # Global average pooling over patches
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
                          padding=(kernel_size - 1) // 2, groups=in_chs, bias=False),
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
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.blocks = self._make_layer(block_module, mid_channels, mid_channels, num_blocks, stride=1, use_eca=use_eca, k_size=k_size)
        
        self.conv_transition = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_transition = nn.BatchNorm2d(mid_channels)
        
        self.conv_shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(mid_channels)
        
        if stride != 1 or in_channels != out_channels: # Downsampling for the entire block if stride > 1
             self.downsample_all = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
             )
        else:
            self.downsample_all = None


        self.conv_final = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)


    def _make_layer(self, block_module, in_planes, planes, num_blocks, stride, use_eca=False, k_size=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            if use_eca:
                layers.append(block_module(in_planes, planes, stride=s, k_size=k_size))
            else:
                 layers.append(block_module(in_planes, planes, stride=s))
            in_planes = planes * block_module.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.downsample_all: # Apply downsampling to input if needed for this CSP stage
            x_shortcut_path = self.downsample_all(x)
            # Re-calculate x for internal paths based on potentially downsampled x
            x_main_path = F.relu(self.bn_shortcut(self.conv_shortcut(x_shortcut_path))) # Process downsampled for main
            x_shortcut_path_internal = F.relu(self.bn1(self.conv1(x_shortcut_path))) # Process downsampled for shortcut
        else:
            x_main_path = F.relu(self.bn_shortcut(self.conv_shortcut(x)))
            x_shortcut_path_internal = F.relu(self.bn1(self.conv1(x)))


        x_main_path = self.blocks(x_main_path)
        x_main_path = F.relu(self.bn_transition(self.conv_transition(x_main_path)))
        
        out = torch.cat((x_main_path, x_shortcut_path_internal), dim=1)
        out = F.relu(self.bn_final(self.conv_final(out)))
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
        self.fc = nn.Linear(512 * block_module.expansion, num_classes)

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
                 deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False):
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNeStCustom, self).__init__()

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
        
        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down, is_first=True) # is_first for AVD in ResNeSt-D
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=True, is_first=False):
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
            layers.append(block(self.inplanes, planes, radix=self.radix,
                                cardinality=self.cardinality, bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first))
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
        super(MBConvBlock, self).__init__()
        self.stride = stride
        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1: # Expansion phase
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Squeeze-and-excitation
        if se_ratio > 0:
            layers.append(SqueezeExcite(hidden_dim, se_ratio=se_ratio))
            
        # Projection phase
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        
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
                 mbconv_expand_ratio=4, transformer_heads=4, transformer_mlp_dim_ratio=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, s0_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(s0_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stage 0 (MBConv)
        self.s0 = self._make_mbconv_stage(s0_channels, s0_channels, s0_blocks, stride=1, expand_ratio=mbconv_expand_ratio)
        
        # Stage 1 (MBConv)
        self.s1 = self._make_mbconv_stage(s0_channels, s1_channels, s1_blocks, stride=2, expand_ratio=mbconv_expand_ratio)
        
        # Stage 2 (Transformer, simplified)
        self.s2_pre_conv = nn.Conv2d(s1_channels, s2_channels, kernel_size=1) # Patch embedding like
        self.s2_transformer = self._make_transformer_stage(s2_channels, s2_blocks, transformer_heads, s2_channels // transformer_heads, s2_channels * transformer_mlp_dim_ratio)
        
        # Stage 3 (Transformer, simplified)
        self.s3_pre_conv = nn.Conv2d(s2_channels, s3_channels, kernel_size=2, stride=2) # Downsampling
        self.s3_transformer = self._make_transformer_stage(s3_channels, s3_blocks, transformer_heads, s3_channels // transformer_heads, s3_channels * transformer_mlp_dim_ratio)

        # Stage 4 (Transformer, simplified)
        self.s4_pre_conv = nn.Conv2d(s3_channels, s4_channels, kernel_size=2, stride=2) # Downsampling
        self.s4_transformer = self._make_transformer_stage(s4_channels, s4_blocks, transformer_heads, s4_channels // transformer_heads, s4_channels * transformer_mlp_dim_ratio)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(s4_channels, num_classes)

    def _make_mbconv_stage(self, in_c, out_c, num_blocks, stride, expand_ratio):
        layers = [MBConvBlock(in_c if i == 0 else out_c, out_c, stride if i == 0 else 1, expand_ratio) for i in range(num_blocks)]
        return nn.Sequential(*layers)

    def _make_transformer_stage(self, dim, num_blocks, heads, dim_head, mlp_dim):
        layers = [TransformerBlock(dim, heads, dim_head, mlp_dim) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.s0(x)
        x = self.s1(x)
        
        # S2
        x = self.s2_pre_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        x = self.s2_transformer(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)

        # S3
        x = self.s3_pre_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.s3_transformer(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)

        # S4
        x = self.s4_pre_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.s4_transformer(x)
        # x = x.transpose(1, 2).reshape(b, c, h, w) # Not needed before pooling if an MLP head takes (B, N, C)
        
        x = torch.mean(x, dim=1) # Global average pooling over sequence length for transformer output

        # x = self.pool(x).flatten(1) # If last stage was conv
        x = self.fc(x)
        return x

# --- HorNet specific blocks (Simplified) ---
class GlobalLocalFilter(nn.Module): # Simplified
    def __init__(self, dim, h=14, w=8): # h, w are example feature map sizes
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        self.proj = nn.Conv2d(dim, dim, 1) # Combined projection
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        
        x2_fft = torch.fft.rfft2(x2, norm='ortho')
        weight_fft = torch.view_as_complex(self.complex_weight)
        x2_filtered = torch.fft.irfft2(x2_fft * weight_fft, s=(x2.size(-2), x2.size(-1)), norm='ortho')
        
        out = torch.cat((x1, x2_filtered), dim=1)
        out = self.bn(self.proj(out))
        return out

class gnConv(nn.Module): # Simplified Recursive Gated Convolution
    def __init__(self, dim, order=5, dw_kernel_size=7, use_filter=True):
        super().__init__()
        self.order = order
        self.dim = dim
        self.use_filter = use_filter

        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1) # For gates and features
        self.simplied_gated_conv = nn.Conv2d(dim, dim, dw_kernel_size, padding=dw_kernel_size//2, groups=dim)
        
        if use_filter:
            self.filter = GlobalLocalFilter(dim) # Simplified filter

        self.pwconv2 = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # Simplified gnConv: using a standard DW conv and optional filter
        # Original gnConv has recursive gating, which is more complex
        
        if self.use_filter:
            x = self.filter(x) # Apply filter first

        # Simplified gating mechanism: split, process one branch, multiply
        x_res = x
        x_gate_feat = self.pwconv1(x)
        x_feat, x_gate = torch.chunk(x_gate_feat, 2, dim=1)
        
        x_conv = self.simplied_gated_conv(x_feat * torch.sigmoid(x_gate)) # Gated DW conv
        
        x_out = self.pwconv2(x_conv)
        x_out = self.bn(x_out)
        x_out = self.act(x_out + x_res) # Add residual
        return x_out

class HorNetBlock(nn.Module):
    def __init__(self, dim, order=5, dw_kernel_size=7, use_filter=True, drop_path=0.):
        super().__init__()
        self.norm = LayerNorm2d(dim) # Using LayerNorm like ConvNeXt
        self.gnconv = gnConv(dim, order, dw_kernel_size, use_filter)
        # drop_path would be here if used
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.gnconv(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        x = input_x + x
        return x

class HorNetCustom(nn.Module): # Simplified HorNet Structure
    def __init__(self, num_classes=100, dims=[64, 128, 256, 512], depths=[2,2,6,2], order=5, dw_k=7, use_filter=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.stages = nn.ModuleList()
        current_dim = dims[0]
        for i in range(len(dims)):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(HorNetBlock(current_dim, order, dw_k, use_filter))
            
            self.stages.append(nn.Sequential(*stage_blocks))
            if i < len(dims) - 1: # Downsampling layer
                self.stages.append(
                    nn.Sequential(
                        LayerNorm2d(current_dim),
                        nn.Conv2d(current_dim, dims[i+1], kernel_size=2, stride=2)
                    )
                )
                current_dim = dims[i+1]
        
        self.norm_head = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage_or_downsample in enumerate(self.stages):
            x = stage_or_downsample(x)
        return self.norm_head(x.mean([-2, -1])) # global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


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

@register_model("segnext_mscan_tiny") # This seems to be MSCANEncoderCustom from before
def segnext_mscan_tiny_custom_builder(num_classes=100, **kwargs):
    # Dims and depths for a "Tiny" variant, e.g. like SegNeXt-T
    return MSCANEncoderCustom(dims=[32, 64, 160, 256], depths=[3, 3, 5, 2], num_classes=num_classes, **kwargs)

@register_model("mlp_mixer_tiny")
def mlp_mixer_tiny_custom_builder(num_classes=100, **kwargs):
    # Default "Tiny" parameters for CIFAR-100 like
    return MLPMixerCustom(image_size=32, patch_size=4, dim=128, depth=8, 
                          token_mlp_dim=256, channel_mlp_dim=512, num_classes=num_classes, **kwargs)

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
    return ResNeStCustom(ResNeStBottleneckD, [3, 4, 6, 3], num_classes=num_classes,
                         radix=2, cardinality=1, bottleneck_width=64,
                         deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False, **kwargs)

@register_model("coatnet_0_custom") # Simplified CoAtNet
def coatnet_0_custom_builder(num_classes=100, **kwargs):
    # Example CoAtNet-0 like structure (very simplified)
    # True CoAtNet-0 has specific channel/block counts per stage from paper
    # This is a placeholder demonstrating the hybrid structure.
    return CoAtNetCustom(num_classes=num_classes, 
                        s0_channels=32, s1_channels=64, s2_channels=128, s3_channels=256, s4_channels=512,
                        s0_blocks=2, s1_blocks=2, s2_blocks=2, s3_blocks=2, s4_blocks=2, 
                        mbconv_expand_ratio=4, transformer_heads=4, transformer_mlp_dim_ratio=2, **kwargs) # Reduced mlp_dim_ratio

@register_model("hornet_tiny_custom") # Simplified HorNet
def hornet_tiny_custom_builder(num_classes=100, **kwargs):
    # Simplified "Tiny" HorNet
    return HorNetCustom(num_classes=num_classes, dims=[32,64,128,256], depths=[2,2,4,2], order=3, dw_k=5, **kwargs)


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
        "mlp_mixer_b16", "coatnet_0_custom", "hornet_tiny_custom"
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