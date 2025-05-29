import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import get_model, get_model_info, MODEL_REGISTRY

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()

class ComparisonExperiment:
    def __init__(self, epochs=15, batch_size=128, learning_rate=0.1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.accelerator = Accelerator()
        
    def _create_data_loaders(self, use_imagenet_norm=False):
        """创建数据加载器"""
        if use_imagenet_norm:
            # ImageNet预训练模型使用的归一化
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            resize_size = 224
        else:
            # CIFAR-100原生归一化
            normalize = transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
            resize_size = 32
        
        transform_train = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resize_size, padding=4) if resize_size == 32 else transforms.RandomCrop(resize_size, padding=28),
            transforms.ToTensor(),
            normalize
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            normalize
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def train_model(self, model, model_name, use_imagenet_norm=False):
        """训练单个模型"""
        print(f"\n开始训练模型: {model_name}")
        
        train_loader, test_loader = self._create_data_loaders(use_imagenet_norm)
        
        # 对预训练模型使用较小的学习率
        if 'timm' in model_name or any(x in model_name for x in ['coatnet', 'cspresnet', 'ghostnet', 'hornet', 'resnest', 'mlp_mixer_b16']):
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate
            
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        
        model, optimizer, train_loader, test_loader = self.accelerator.prepare(
            model, optimizer, train_loader, test_loader
        )
        
        results = {
            'model_name': model_name,
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_acc_top5': [],
            'best_acc': 0,
            'best_acc_top5': 0,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
            'total_time': 0,
            'learning_rate': lr
        }
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                self.accelerator.backward(loss)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    self.accelerator.print(
                        f'Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%'
                    )
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # 测试阶段
            model.eval()
            test_loss = 0
            test_correct = 0
            test_correct_top5 = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
                    
                    # Top-5准确率
                    _, top5_pred = outputs.topk(5, 1, True, True)
                    top5_pred = top5_pred.t()
                    test_correct_top5 += top5_pred.eq(targets.view(1, -1).expand_as(top5_pred)).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = 100. * test_correct / test_total
            test_acc_top5 = 100. * test_correct_top5 / test_total
            
            # 记录结果
            results['epochs'].append(epoch + 1)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            results['test_acc_top5'].append(test_acc_top5)
            
            if test_acc > results['best_acc']:
                results['best_acc'] = test_acc
                results['best_acc_top5'] = test_acc_top5
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            self.accelerator.print(
                f'Epoch {epoch+1}/{self.epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                f'Test Top5: {test_acc_top5:.2f}%, Time: {epoch_time:.1f}s'
            )
        
        results['total_time'] = time.time() - start_time
        
        self.accelerator.print(f"模型 {model_name} 训练完成!")
        self.accelerator.print(f"最佳准确率: {results['best_acc']:.2f}%, 参数量: {results['parameters']:.2f}M")
        
        return results

class ArchitectureComparison:
    """不同架构类型的对比实验"""
    
    @staticmethod
    def get_model_groups():
        """按技术类型分组模型"""
        groups = {
            '基础卷积网络': [
                'resnet_20',
                'resnet_32'
            ],
            '注意力机制': [
                'eca_resnet_20',
                'segnext_mscan_tiny'
            ],
            '轻量化设计': [
                'ghost_resnet_20',
                'ghostnet_100'
            ],
            '现代化架构': [
                'convnext_tiny',
                'convnext_tiny_timm'
            ],
            '混合架构': [
                'coatnet_0',
                'resnest50d'
            ],
            'MLP架构': [
                'mlp_mixer_tiny',
                'mlp_mixer_b16'
            ]
        }
        return groups

class EfficiencyComparison:
    """效率对比实验"""
    
    @staticmethod
    def get_efficiency_models():
        """获取用于效率对比的模型"""
        return [
            'resnet_20',           # 基准
            'eca_resnet_20',       # 注意力增强
            'ghost_resnet_20',     # 轻量化
            'convnext_tiny',       # 现代架构
            'ghostnet_100',        # 工业级轻量化
        ]

class PretrainedVsFromScratch:
    """预训练 vs 从头训练对比"""
    
    @staticmethod
    def get_comparison_pairs():
        """获取对比配对"""
        return [
            ('convnext_tiny', 'convnext_tiny_timm'),      # 自实现 vs timm预训练
            ('mlp_mixer_tiny', 'mlp_mixer_b16'),          # 小模型 vs 预训练大模型
        ]

def run_architecture_comparison():
    """运行架构对比实验"""
    experiment = ComparisonExperiment(epochs=15)
    groups = ArchitectureComparison.get_model_groups()
    
    all_results = {}
    
    for group_name, model_names in groups.items():
        print(f"\n{'='*60}")
        print(f"开始{group_name}组对比实验")
        print(f"{'='*60}")
        
        group_results = {}
        
        for model_name in model_names:
            if model_name not in MODEL_REGISTRY:
                print(f"跳过不存在的模型: {model_name}")
                continue
                
            try:
                model = get_model(model_name)
                
                # 判断是否需要ImageNet归一化
                use_imagenet_norm = any(x in model_name for x in ['timm', 'coatnet', 'cspresnet', 'ghostnet', 'hornet', 'resnest', 'mlp_mixer_b16'])
                
                result = experiment.train_model(model, model_name, use_imagenet_norm)
                group_results[model_name] = result
                
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")
                group_results[model_name] = {'error': str(e)}
        
        all_results[group_name] = group_results
        
        # 保存组结果
        results_dir = Path('logs/comparison_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        group_file = results_dir / f'{group_name.replace(" ", "_")}.json'
        with open(group_file, 'w', encoding='utf-8') as f:
            json.dump(group_results, f, indent=2, ensure_ascii=False)
        
        # 打印组结果汇总
        print(f"\n{group_name}组结果汇总:")
        print("-" * 60)
        for model_name, result in group_results.items():
            if 'error' not in result:
                print(f"{model_name:<25}: {result['best_acc']:6.2f}% "
                      f"(参数: {result['parameters']:5.2f}M, 时间: {result['total_time']:6.1f}s)")
            else:
                print(f"{model_name:<25}: 失败")
    
    # 保存总结果
    summary_file = results_dir / 'all_comparison_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results

def run_efficiency_comparison():
    """运行效率对比实验"""
    experiment = ComparisonExperiment(epochs=10)  # 较短的训练用于效率测试
    model_names = EfficiencyComparison.get_efficiency_models()
    
    results = {}
    
    print(f"\n{'='*60}")
    print("效率对比实验")
    print(f"{'='*60}")
    
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            continue
            
        try:
            model = get_model(model_name)
            
            # 测量前向传播时间
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(32, 3, 32, 32)  # 小批量测试
                start_time = time.time()
                for _ in range(100):  # 多次测试取平均
                    _ = model(dummy_input)
                forward_time = (time.time() - start_time) / 100
            
            # 训练测试
            train_result = experiment.train_model(model, model_name, False)
            train_result['forward_time_ms'] = forward_time * 1000
            train_result['param_efficiency'] = train_result['best_acc'] / train_result['parameters']
            train_result['time_efficiency'] = train_result['best_acc'] / train_result['total_time']
            
            results[model_name] = train_result
            
        except Exception as e:
            print(f"模型 {model_name} 效率测试失败: {e}")
            results[model_name] = {'error': str(e)}
    
    # 保存效率结果
    results_dir = Path('logs/comparison_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    efficiency_file = results_dir / 'efficiency_comparison.json'
    with open(efficiency_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印效率汇总
    print(f"\n效率对比结果:")
    print("-" * 80)
    print(f"{'模型名称':<20} {'准确率%':<8} {'参数M':<8} {'训练时间s':<10} {'前向时间ms':<12} {'参数效率':<10} {'时间效率':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name:<20} {result['best_acc']:<8.2f} {result['parameters']:<8.2f} "
                  f"{result['total_time']:<10.1f} {result['forward_time_ms']:<12.3f} "
                  f"{result['param_efficiency']:<10.2f} {result['time_efficiency']:<10.4f}")
    
    return results

def create_comparison_plots(results):
    """创建对比图表"""
    
    # 确保assets目录存在
    Path('assets').mkdir(exist_ok=True)
    
    # 1. 架构类型性能对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (group_name, group_results) in enumerate(results.items()):
        if i >= 6:  # 最多6个子图
            break
            
        ax = axes[i]
        
        # 提取有效结果
        valid_results = {k: v for k, v in group_results.items() if 'error' not in v}
        
        if not valid_results:
            ax.text(0.5, 0.5, '无有效结果', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group_name)
            continue
        
        model_names = list(valid_results.keys())
        accuracies = [result['best_acc'] for result in valid_results.values()]
        parameters = [result['parameters'] for result in valid_results.values()]
        
        # 气泡图：x轴参数量，y轴准确率，气泡大小表示训练时间
        sizes = [result['total_time'] / 10 for result in valid_results.values()]  # 缩放
        
        scatter = ax.scatter(parameters, accuracies, s=sizes, alpha=0.6, c=range(len(model_names)), cmap='viridis')
        
        # 添加标签
        for j, name in enumerate(model_names):
            ax.annotate(name.replace('_', '\n'), (parameters[j], accuracies[j]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('参数量 (M)')
        ax.set_ylabel('准确率 (%)')
        ax.set_title(group_name)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(results), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('assets/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 总体性能排名图
    all_models = []
    for group_results in results.values():
        for model_name, result in group_results.items():
            if 'error' not in result:
                all_models.append({
                    'name': model_name,
                    'accuracy': result['best_acc'],
                    'parameters': result['parameters'],
                    'time': result['total_time']
                })
    
    if all_models:
        # 按准确率排序
        all_models.sort(key=lambda x: x['accuracy'], reverse=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 准确率排名
        names = [model['name'].replace('_', '\n') for model in all_models]
        accuracies = [model['accuracy'] for model in all_models]
        
        bars = ax1.bar(range(len(names)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('模型准确率排名', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 效率散点图
        parameters = [model['parameters'] for model in all_models]
        times = [model['time'] for model in all_models]
        
        scatter = ax2.scatter(parameters, accuracies, s=[t/10 for t in times], 
                             alpha=0.6, c=range(len(all_models)), cmap='viridis')
        ax2.set_xlabel('参数量 (M)')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('参数效率分析')
        ax2.grid(True, alpha=0.3)
        
        # 添加标签
        for i, model in enumerate(all_models):
            ax2.annotate(model['name'].replace('_', '\n'), 
                        (parameters[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('assets/overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    print("开始运行对比实验...")
    
    # 1. 架构对比实验
    print("\n" + "="*80)
    print("1. 架构类型对比实验")
    print("="*80)
    
    architecture_results = run_architecture_comparison()
    
    # 2. 效率对比实验
    print("\n" + "="*80)
    print("2. 效率对比实验")
    print("="*80)
    
    efficiency_results = run_efficiency_comparison()
    
    # 3. 生成对比图表
    print("\n" + "="*80)
    print("3. 生成对比图表")
    print("="*80)
    
    create_comparison_plots(architecture_results)
    
    print("\n对比实验完成!")
    print("结果保存在 logs/comparison_results/ 目录")
    print("图表保存在 assets/ 目录")

if __name__ == "__main__":
    main() 