import torch
import torch.nn as nn
from model import get_model, get_model_info, count_parameters

def test_coatnet_cifar_opt():
    print("测试CoAtNet-CIFAROpt模型实现")
    print("=" * 50)
    
    # 测试两个版本的模型
    model_names = ['coatnet_cifar_opt', 'coatnet_cifar_opt_large_stem']
    
    for model_name in model_names:
        print(f"\n测试模型: {model_name}")
        print("-" * 30)
        
        try:
            # 创建模型
            model = get_model(model_name, num_classes=100)
            
            # 获取模型信息
            model_info = get_model_info(model, model_name)
            print(f"模型信息: {model_info}")
            
            # 测试前向传播
            model.eval()
            batch_size = 4
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"输入形状: {input_tensor.shape}")
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
            # 验证输出形状
            expected_shape = (batch_size, 100)
            if output.shape == expected_shape:
                print("✓ 输出形状正确")
            else:
                print(f"✗ 输出形状错误，期望 {expected_shape}，得到 {output.shape}")
            
            # 测试梯度传播
            model.train()
            criterion = nn.CrossEntropyLoss()
            targets = torch.randint(0, 100, (batch_size,))
            
            output = model(input_tensor)
            loss = criterion(output, targets)
            loss.backward()
            
            # 检查梯度
            has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            if has_grad:
                print("✓ 梯度传播正常")
            else:
                print("✗ 梯度传播异常")
            
            print(f"损失值: {loss.item():.4f}")
            
        except Exception as e:
            print(f"✗ 模型测试失败: {e}")
        
        print("")

def test_eca_components():
    print("测试ECA组件")
    print("=" * 50)
    
    from model import ECALayer, ECAMBConvBlock, ECATransformerBlock
    
    # 测试ECALayer
    print("测试ECALayer...")
    try:
        eca = ECALayer(channels=64, k_size=3)
        x = torch.randn(2, 64, 8, 8)
        out = eca(x)
        if out.shape == x.shape:
            print("✓ ECALayer测试通过")
        else:
            print(f"✗ ECALayer输出形状错误: {out.shape} vs {x.shape}")
    except Exception as e:
        print(f"✗ ECALayer测试失败: {e}")
    
    # 测试ECAMBConvBlock
    print("测试ECAMBConvBlock...")
    try:
        block = ECAMBConvBlock(inp=64, oup=96, stride=2, expand_ratio=4, k_size=3)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        expected_shape = (2, 96, 8, 8)
        if out.shape == expected_shape:
            print("✓ ECAMBConvBlock测试通过")
        else:
            print(f"✗ ECAMBConvBlock输出形状错误: {out.shape} vs {expected_shape}")
    except Exception as e:
        print(f"✗ ECAMBConvBlock测试失败: {e}")
    
    # 测试ECATransformerBlock
    print("测试ECATransformerBlock...")
    try:
        block = ECATransformerBlock(dim=384, heads=8, dim_head=48, mlp_dim=1536, dropout=0.1)
        x = torch.randn(2, 4, 384)  # (batch, seq_len, dim)
        out = block(x)
        if out.shape == x.shape:
            print("✓ ECATransformerBlock测试通过")
        else:
            print(f"✗ ECATransformerBlock输出形状错误: {out.shape} vs {x.shape}")
    except Exception as e:
        print(f"✗ ECATransformerBlock测试失败: {e}")

def compare_with_baseline():
    print("与基线模型比较")
    print("=" * 50)
    
    # 比较模型参数量和FLOPs
    models_to_compare = [
        'coatnet_0',
        'coatnet_cifar_opt',
        'coatnet_cifar_opt_large_stem'
    ]
    
    print(f"{'模型名称':<30} {'参数量(M)':<15} {'FLOPs(G)':<15}")
    print("-" * 60)
    
    for model_name in models_to_compare:
        try:
            model = get_model(model_name, num_classes=100)
            params = count_parameters(model)
            
            # 简单的FLOPs估算（基于参数量的粗略估计）
            # 实际应用中建议使用thop或fvcore库进行精确计算
            flops_estimate = params * 2 * 32 * 32 / 1e9  # 粗略估计
            
            print(f"{model_name:<30} {params:<15.2f} {flops_estimate:<15.2f}")
            
        except Exception as e:
            print(f"{model_name:<30} 错误: {e}")

def test_training_components():
    print("测试训练组件")
    print("=" * 50)
    
    # 导入训练相关组件
    import sys
    sys.path.append('.')
    from train_coatnet_cifar_opt import MixUpCutMix, LabelSmoothingCrossEntropy
    
    # 测试MixUpCutMix
    print("测试MixUpCutMix...")
    try:
        mixup_cutmix = MixUpCutMix(mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        
        mixed_x, mixed_y = mixup_cutmix((x, y))
        
        if mixed_x.shape == x.shape and len(mixed_y) == 3:
            print("✓ MixUpCutMix测试通过")
        else:
            print(f"✗ MixUpCutMix测试失败")
    except Exception as e:
        print(f"✗ MixUpCutMix测试失败: {e}")
    
    # 测试LabelSmoothingCrossEntropy
    print("测试LabelSmoothingCrossEntropy...")
    try:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        loss = criterion(logits, targets)
        
        if loss.item() > 0:
            print("✓ LabelSmoothingCrossEntropy测试通过")
        else:
            print("✗ LabelSmoothingCrossEntropy测试失败")
    except Exception as e:
        print(f"✗ LabelSmoothingCrossEntropy测试失败: {e}")

def main():
    print("CoAtNet-CIFAROpt 综合测试")
    print("=" * 70)
    
    # 执行各项测试
    test_coatnet_cifar_opt()
    test_eca_components()
    compare_with_baseline()
    test_training_components()
    
    print("测试完成！")

if __name__ == '__main__':
    main() 