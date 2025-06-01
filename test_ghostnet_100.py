import torch
from src.model import get_model, count_parameters
from src.dataset import get_dataloaders

def test_ghostnet_100():
    print("测试 GhostNet-100 模型")
    print("="*50)
    
    try:
        print("1. 创建模型...")
        model = get_model(
            model_name='ghostnet_100',
            num_classes=100,
            pretrained_timm=True
        )
        
        param_count = count_parameters(model)
        print(f"   模型参数量: {param_count:.2f}M")
        
        print("2. 测试前向传播...")
        dummy_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        
        print("3. 创建数据加载器...")
        train_loader, test_loader = get_dataloaders(
            batch_size=64,
            use_imagenet_norm=True,
            num_workers=2
        )
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        print("4. 测试一个训练批次...")
        model.train()
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            print(f"   批次大小: {inputs.shape[0]}")
            print(f"   损失值: {loss.item():.4f}")
            break
        
        print("\n✅ 所有测试通过! GhostNet-100 模型可以正常训练")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ghostnet_100() 