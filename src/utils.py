import torch
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, model_name="Model"):
    total_params = count_parameters(model)
    print(f'{model_name} 总参数量: {total_params:,}')
    print(f'{model_name} 总参数量: {total_params/1e6:.2f}M')
    
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')

def plot_training_curves(history, model_name, save_path=None):
    if save_path is None:
        save_path = f'logs/{model_name}_training_curves.png'
    
    os.makedirs('logs', exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['test_losses'], 'r-', label='测试损失')
    ax1.set_title(f'{model_name} 损失曲线')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_accs'], 'b-', label='训练准确率')
    ax2.plot(epochs, history['test_accs'], 'r-', label='测试准确率')
    ax2.set_title(f'{model_name} 准确率曲线')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    last_n = min(20, len(epochs))
    ax3.plot(epochs[-last_n:], history['train_losses'][-last_n:], 'b-', label='训练损失')
    ax3.plot(epochs[-last_n:], history['test_losses'][-last_n:], 'r-', label='测试损失')
    ax3.set_title(f'{model_name} 损失曲线（最后{last_n}轮）')
    ax3.set_xlabel('轮次')
    ax3.set_ylabel('损失')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(epochs[-last_n:], history['train_accs'][-last_n:], 'b-', label='训练准确率')
    ax4.plot(epochs[-last_n:], history['test_accs'][-last_n:], 'r-', label='测试准确率')
    ax4.set_title(f'{model_name} 准确率曲线（最后{last_n}轮）')
    ax4.set_xlabel('轮次')
    ax4.set_ylabel('准确率 (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'训练曲线已保存到: {save_path}')

def save_results(history, model_name, save_path='logs/results.txt'):
    os.makedirs('logs', exist_ok=True)
    with open(save_path, 'a') as f:
        f.write(f'\n{model_name} 训练结果:\n')
        f.write(f'最佳测试准确率: {history["best_acc"]:.2f}%\n')
        f.write(f'最终训练损失: {history["train_losses"][-1]:.4f}\n')
        f.write(f'最终测试损失: {history["test_losses"][-1]:.4f}\n')
        f.write(f'最终训练准确率: {history["train_accs"][-1]:.2f}%\n')
        f.write(f'最终测试准确率: {history["test_accs"][-1]:.2f}%\n')
        f.write('-' * 50 + '\n')

def compare_models(results_dict, save_path='logs/model_comparison.png'):
    os.makedirs('logs', exist_ok=True)
    
    models = list(results_dict.keys())
    best_accs = [results_dict[model]['best_acc'] for model in models]
    final_accs = [results_dict[model]['test_accs'][-1] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = range(len(models))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], best_accs, width, label='最佳准确率', alpha=0.8)
    ax1.bar([i + width/2 for i in x], final_accs, width, label='最终准确率', alpha=0.8)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('模型性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (best, final) in enumerate(zip(best_accs, final_accs)):
        ax1.text(i - width/2, best + 0.5, f'{best:.1f}%', ha='center', fontsize=10)
        ax1.text(i + width/2, final + 0.5, f'{final:.1f}%', ha='center', fontsize=10)
    
    for model in models:
        epochs = range(1, len(results_dict[model]['test_accs']) + 1)
        ax2.plot(epochs, results_dict[model]['test_accs'], label=model, linewidth=2)
    
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('测试准确率 (%)')
    ax2.set_title('测试准确率变化曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'模型对比图已保存到: {save_path}')

def save_comparison_results(results_dict, save_path='logs/comparison_results.txt'):
    os.makedirs('logs', exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('CIFAR-100 分类模型性能对比结果\n')
        f.write('='*60 + '\n\n')
        
        f.write(f'{"模型名称":<20} {"最佳准确率":<12} {"最终准确率":<12} {"参数量":<12}\n')
        f.write('-'*60 + '\n')
        
        for model_name, history in results_dict.items():
            best_acc = history['best_acc']
            final_acc = history['test_accs'][-1]
            f.write(f'{model_name:<20} {best_acc:<12.2f} {final_acc:<12.2f}\n')
        
        f.write('\n' + '='*60 + '\n')
        f.write('详细分析:\n')
        f.write('='*60 + '\n')
        
        best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['best_acc'])
        f.write(f'最佳模型: {best_model} (准确率: {results_dict[best_model]["best_acc"]:.2f}%)\n')
        
        baseline_acc = results_dict.get('ResNet-20', {}).get('best_acc', 0)
        if baseline_acc > 0:
            f.write(f'\n相对于基线ResNet-20的提升:\n')
            for model_name, history in results_dict.items():
                if model_name != 'ResNet-20':
                    improvement = history['best_acc'] - baseline_acc
                    f.write(f'{model_name}: {improvement:+.2f}%\n')
    
    print(f'对比结果已保存到: {save_path}') 