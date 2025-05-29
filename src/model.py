import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
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

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str) -> Callable[..., Callable[..., nn.Module]]:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = builder
        return builder
    return decorator

@register_model("resnet_20")
def resnet20_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs=kwargs)

@register_model("resnet_32")
def resnet32_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs=kwargs)

@register_model("resnet_56")
def resnet56_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, block_kwargs=kwargs)

@register_model("eca_resnet_20")
def eca_resnet20_builder(num_classes=100, k_size=3, **kwargs):
    return ResNet(ECABasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs={'k_size': k_size, **kwargs})

@register_model("eca_resnet_32")
def eca_resnet32_builder(num_classes=100, k_size=3, **kwargs):
    return ResNet(ECABasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs={'k_size': k_size, **kwargs})

@register_model("ghost_resnet_20")
def ghost_resnet20_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs={'ratio': ratio, **kwargs})

@register_model("ghost_resnet_32")
def ghost_resnet32_builder(num_classes=100, ratio=2, **kwargs):
    return ResNet(GhostBasicBlock, [5, 5, 5], num_classes=num_classes, block_kwargs={'ratio': ratio, **kwargs})

@register_model("convnext_tiny")
def convnext_tiny_custom_builder(num_classes=100, **kwargs):
    return ConvNeXtCustom(depths=[2,2,6,2], dims=[32,64,128,256], num_classes=num_classes, **kwargs)

@register_model("segnext_mscan_tiny")
def segnext_mscan_tiny_custom_builder(num_classes=100, **kwargs):
    return MSCANEncoderCustom(dims=[32,64,128], depths=[2,2,3], num_classes=num_classes, **kwargs)

@register_model("mlp_mixer_tiny")
def mlp_mixer_tiny_custom_builder(num_classes=100, **kwargs):
    return MLPMixerCustom(image_size=32, patch_size=4, dim=128, depth=4, 
                          token_mlp_dim=128, channel_mlp_dim=256,
                          num_classes=num_classes, **kwargs)

TIMM_NAME_MAP = {
    "convnext_tiny_timm": "convnext_tiny",
    "coatnet_0": "coatnet_0",
    "cspresnet50": "cspresnet50",
    "ghostnet_100": "ghostnet_100",
    "hornet_tiny": "hornet_tiny_7x7",
    "resnest50d": "resnest50d",
    "mlp_mixer_b16": "mixer_b16_224_in21k"
}

def get_model(model_name: str, num_classes: int = 100, pretrained_timm: bool = False, **kwargs: Any) -> nn.Module:
    if pretrained_timm and model_name in TIMM_NAME_MAP:
        # For timm models, pretrained=True implies ImageNet pretraining.
        # kwargs might include specific timm features if necessary
        model = timm.create_model(
            TIMM_NAME_MAP[model_name],
            pretrained=True,
            num_classes=num_classes,
            **kwargs.get('timm_extra_args', {}) # Pass any extra args for timm
        )
        # If 'drop_path_rate' is in kwargs, ensure it's applied if the model supports it
        if 'drop_path_rate' in kwargs and hasattr(model, 'drop_path_rate'):
            model.drop_path_rate = kwargs['drop_path_rate']
        return model
    
    builder = MODEL_REGISTRY.get(model_name)
    if builder:
        # Pass only relevant kwargs to the builder
        # This requires builders to be robust to extra kwargs or for us to filter them
        # For simplicity here, we pass all, assuming builders handle them or use **kwargs
        return builder(num_classes=num_classes, **kwargs)
    
    raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY or TIMM_NAME_MAP for the given configuration.")

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model_info(model: nn.Module, model_name: str, pretrained_timm: bool = False) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "parameters_M": count_parameters(model),
        "is_timm_pretrained": pretrained_timm
    }

if __name__ == '__main__':
    model_names_to_test = [
        "resnet_20", "resnet_32", "resnet_56",
        "eca_resnet_20", "eca_resnet_32",
        "ghost_resnet_20", "ghost_resnet_32",
        "convnext_tiny", 
        "segnext_mscan_tiny",
        "mlp_mixer_tiny",
        "convnext_tiny_timm", 
        "coatnet_0", 
        "cspresnet50",
        "ghostnet_100",
        "hornet_tiny",
        "resnest50d",
        "mlp_mixer_b16",
    ]

    for name in model_names_to_test:
        print(f"--- Testing model: {name} ---")
        try:
            if name in ["eca_resnet_20", "eca_resnet_32"]:
                model_instance = get_model(name, num_classes=100, pretrained_timm=(name.endswith("_timm")), k_size=3)
            elif name in ["ghost_resnet_20", "ghost_resnet_32"]:
                model_instance = get_model(name, num_classes=100, pretrained_timm=(name.endswith("_timm")), ratio=2)
            else:
                model_instance = get_model(name, num_classes=100, pretrained_timm=(name.endswith("_timm") or name in TIMM_NAME_MAP))
            
            dummy_input = torch.randn(2, 3, 32, 32)
            output = model_instance(dummy_input)
            print(f"Model: {name}, Output shape: {output.shape}, Params (M): {count_parameters(model_instance):.2f}")
            
            if name == "eca_resnet_20":
                 model_instance_k5 = get_model(name, num_classes=100, k_size=5)
                 output_k5 = model_instance_k5(dummy_input)
                 print(f"Model: {name} (k_size=5), Output shape: {output_k5.shape}, Params (M): {count_parameters(model_instance_k5):.2f}")

        except Exception as e:
            print(f"Error testing model {name}: {e}")
            import traceback
            traceback.print_exc()

# Cleanup old functions if they are fully replaced by the registry
# The functions like resnet20(), eca_resnet20() defined globally are now replaced by builders
# and called via get_model.

# Remove or comment out old separate model functions if `get_model` and registry are comprehensive
# def resnet20(): ... (old)
# def eca_resnet20(): ... (old)
# etc.

# Remove create_timm_model as its logic is in get_model
# def create_timm_model(model_name: str, num_classes: int = 100, pretrained: bool = True): (old)

# Remove old get_model and get_model_info if fully replaced
# def get_model(model_name: str): (old, different signature)
# def get_model_info(model_name: str): (old) 