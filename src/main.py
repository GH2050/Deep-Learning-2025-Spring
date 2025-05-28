import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import resnet20, eca_resnet20, ghost_resnet20, convnext_tiny
from dataset import get_dataloaders
from train import train_model
from utils import print_model_info, plot_training_curves, save_results, compare_models, save_comparison_results

def main():
    print(f'PyTorch 版本: {torch.__version__}')
    print('CIFAR-100 分类项目 - 精简版ResNet与先进架构对比实验')
    print('='*60)
    
    batch_size = 128
    epochs = 50
    learning_rate = 0.1
    
    print(f'实验配置:')
    print(f'  批次大小: {batch_size}')
    print(f'  训练轮次: {epochs}')
    print(f'  初始学习率: {learning_rate}')
    print('='*60)
    
    print('加载CIFAR-100数据集...')
    trainloader, testloader = get_dataloaders(batch_size=batch_size, num_workers=0)
    print(f'训练集大小: {len(trainloader.dataset)}')
    print(f'测试集大小: {len(testloader.dataset)}')
    print('='*60)
    
    models_to_train = [
        ('ResNet-20', resnet20()),
        ('ECA-ResNet-20', eca_resnet20()),
        ('Ghost-ResNet-20', ghost_resnet20()),
        ('ConvNeXt-Tiny', convnext_tiny())
    ]
    
    results = {}
    
    for model_name, model in models_to_train:
        print(f'\n开始训练 {model_name}...')
        print('-'*40)
        
        print_model_info(model, model_name)
        
        history = train_model(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            epochs=epochs,
            lr=learning_rate,
            model_name=model_name.lower().replace('-', '_')
        )
        
        results[model_name] = history
        
        plot_training_curves(history, model_name)
        save_results(history, model_name)
        
        print(f'{model_name} 训练完成！最佳准确率: {history["best_acc"]:.2f}%')
        print('='*60)
    
    print('\n所有模型训练完成！生成对比分析...')
    
    compare_models(results)
    save_comparison_results(results)
    
    print('\n实验结果总结:')
    print('-'*40)
    for model_name, history in results.items():
        print(f'{model_name:<20}: {history["best_acc"]:.2f}%')
    
    best_model = max(results.keys(), key=lambda x: results[x]['best_acc'])
    print(f'\n最佳模型: {best_model} ({results[best_model]["best_acc"]:.2f}%)')
    
    baseline_acc = results.get('ResNet-20', {}).get('best_acc', 0)
    if baseline_acc > 0:
        print(f'\n相对基线ResNet-20的提升:')
        for model_name, history in results.items():
            if model_name != 'ResNet-20':
                improvement = history['best_acc'] - baseline_acc
                print(f'  {model_name}: {improvement:+.2f}%')
    
    print('\n所有结果和图表已保存到 logs/ 目录')

if __name__ == '__main__':
    main() 