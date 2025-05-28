import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import resnet20, eca_resnet20
from train_fast import train_fast
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    print("加载CIFAR-100数据集...")
    trainloader, testloader = get_dataloaders(batch_size=256, num_workers=4)
    print(f"训练集: {len(trainloader)} batches, 测试集: {len(testloader)} batches")
    
    models_config = [
        ('resnet_20', resnet20()),
        ('eca_resnet_20', eca_resnet20()),
    ]
    
    results = {}
    
    for model_name, model in models_config:
        print(f"\n{'='*50}")
        print(f"开始训练 {model_name}")
        print(f"{'='*50}")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {param_count/1e6:.2f}M")
        
        model = model.to(device)
        
        history, best_acc = train_fast(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=15,
            lr=0.1,
            model_name=model_name
        )
        
        results[model_name] = {
            'history': history,
            'best_acc': best_acc,
            'params': param_count
        }
        
        print(f"{model_name} 训练完成! 最佳准确率: {best_acc:.2f}%")
    
    print(f"\n{'='*50}")
    print("训练结果汇总:")
    print(f"{'='*50}")
    for name, result in results.items():
        print(f"{name}: {result['best_acc']:.2f}% (参数: {result['params']/1e6:.2f}M)")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['history']['train_loss'], label=f'{name} 训练')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        plt.plot(result['history']['test_loss'], label=f'{name} 测试')
    plt.title('测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        plt.plot(result['history']['train_acc'], label=f'{name} 训练')
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for name, result in results.items():
        plt.plot(result['history']['test_acc'], label=f'{name} 测试')
    plt.title('测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/training_comparison_fast.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 logs/training_comparison_fast.png")
    
    with open('logs/results_fast.txt', 'w') as f:
        f.write("快速训练结果汇总\n")
        f.write("="*30 + "\n")
        for name, result in results.items():
            f.write(f"{name}: {result['best_acc']:.2f}% (参数: {result['params']/1e6:.2f}M)\n")

if __name__ == "__main__":
    main() 