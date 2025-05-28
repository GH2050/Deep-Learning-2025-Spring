import torch
import sys
import os

sys.path.append('./src')

from model import resnet20, eca_resnet20, ghost_resnet20, convnext_tiny
from dataset import get_dataloaders
from utils import print_model_info

def test_models():
    print("测试模型实例化...")
    
    models = [
        ('ResNet-20', resnet20()),
        ('ECA-ResNet-20', eca_resnet20()),
        ('Ghost-ResNet-20', ghost_resnet20()),
        ('ConvNeXt-Tiny', convnext_tiny())
    ]
    
    for name, model in models:
        try:
            print_model_info(model, name)
            
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                y = model(x)
            print(f"✓ {name} 前向传播测试通过")
            print()
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
            return False
    
    return True

def test_dataloader():
    print("测试数据加载器...")
    try:
        trainloader, testloader = get_dataloaders(batch_size=4, num_workers=0)
        
        batch = next(iter(trainloader))
        print(f"训练批次形状: {batch['image'].shape}, {batch['label'].shape}")
        
        batch = next(iter(testloader))
        print(f"测试批次形状: {batch['image'].shape}, {batch['label'].shape}")
        
        print("✓ 数据加载器测试通过")
        return True
    except Exception as e:
        print(f"✗ 数据加载器失败: {e}")
        return False

if __name__ == '__main__':
    print("开始设置验证...")
    print("="*50)
    
    if test_models() and test_dataloader():
        print("="*50)
        print("✓ 所有测试通过！可以开始训练。")
    else:
        print("="*50)
        print("✗ 测试失败，请检查代码。") 