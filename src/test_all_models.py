import torch
import sys
sys.path.append('src')

from src.model import MODEL_REGISTRY, get_model_info

def test_model(model_name):
    """测试单个模型"""
    print(f"\n测试模型: {model_name}")
    print("-" * 50)
    
    try:
        # 获取模型信息
        model_info = get_model_info(model_name)
        model = model_info['model']
        
        print(f"参数量: {model_info['parameters_M']:.2f}M ({model_info['parameters']} 参数)")
        
        # 创建测试输入
        # CIFAR-100模型使用32x32输入，timm预训练模型使用224x224
        if 'timm' in model_name or model_name in ['coatnet_0', 'cspresnet50', 'ghostnet_100', 'hornet_tiny', 'resnest50d', 'mlp_mixer_b16']:
            test_input = torch.randn(2, 3, 224, 224)
        else:
            test_input = torch.randn(2, 3, 32, 32)
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输入尺寸: {list(test_input.shape)}")
        print(f"输出尺寸: {list(output.shape)}")
        print(f"输出类别数: {output.shape[1]}")
        
        # 验证输出
        assert output.shape[0] == 2, f"批量维度错误: {output.shape[0]} != 2"
        assert output.shape[1] == 100, f"类别数错误: {output.shape[1]} != 100"
        
        print("✅ 模型测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """测试所有模型"""
    print("开始测试所有模型...")
    print("=" * 80)
    
    success_count = 0
    total_count = 0
    failed_models = []
    
    for model_name in MODEL_REGISTRY.keys():
        total_count += 1
        if test_model(model_name):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    print(f"\n{'='*80}")
    print("测试结果汇总:")
    print(f"{'='*80}")
    print(f"总模型数: {total_count}")
    print(f"成功模型数: {success_count}")
    print(f"失败模型数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if failed_models:
        print(f"\n失败的模型:")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\n可用模型列表:")
    for i, model_name in enumerate(MODEL_REGISTRY.keys(), 1):
        print(f"{i:2d}. {model_name}")

if __name__ == "__main__":
    test_model("mlp_mixer_tiny") 