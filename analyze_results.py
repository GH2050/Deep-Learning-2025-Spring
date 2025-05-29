import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# 支持中文
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载所有模型的训练结果"""
    results_dir = Path('logs/results')
    all_results = {}
    
    # 加载汇总文件
    summary_file = results_dir / 'all_models_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    
    # 加载单独的结果文件
    for result_file in results_dir.glob('*_results.json'):
        if 'summary' not in result_file.name:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                model_name = result['model_name']
                all_results[model_name] = result
    
    return all_results

def create_summary_table(results):
    """创建结果汇总表"""
    data = []
    
    for model_name, result in results.items():
        if 'error' not in result and 'best_acc' in result:
            data.append({
                '模型名称': model_name,
                '最佳准确率(%)': result['best_acc'],
                'Top5准确率(%)': result.get('best_acc_top5', 0),
                '参数量(M)': result['parameters'],
                '训练时间(s)': result['total_time'],
                '参数效率': result['best_acc'] / result['parameters'] if result['parameters'] > 0 else 0
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('最佳准确率(%)', ascending=False)
    return df

def plot_accuracy_comparison(results):
    """绘制准确率对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取数据
    model_names = []
    top1_accs = []
    top5_accs = []
    
    for model_name, result in results.items():
        if 'error' not in result and 'best_acc' in result:
            model_names.append(model_name.replace('_', '\n'))
            top1_accs.append(result['best_acc'])
            top5_accs.append(result.get('best_acc_top5', 0))
    
    # 按Top1准确率排序
    sorted_data = sorted(zip(model_names, top1_accs, top5_accs), key=lambda x: x[1], reverse=True)
    model_names, top1_accs, top5_accs = zip(*sorted_data)
    
    # Top1准确率柱状图
    bars1 = ax1.bar(range(len(model_names)), top1_accs, color='skyblue', alpha=0.7)
    ax1.set_title('Top-1 准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars1, top1_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Top5准确率柱状图
    bars2 = ax2.bar(range(len(model_names)), top5_accs, color='lightcoral', alpha=0.7)
    ax2.set_title('Top-5 准确率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars2, top5_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('assets/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_analysis(results):
    """绘制效率分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取数据
    model_names = []
    accuracies = []
    parameters = []
    train_times = []
    
    for model_name, result in results.items():
        if 'error' not in result and 'best_acc' in result:
            model_names.append(model_name)
            accuracies.append(result['best_acc'])
            parameters.append(result['parameters'])
            train_times.append(result['total_time'])
    
    # 准确率 vs 参数量散点图
    scatter1 = ax1.scatter(parameters, accuracies, s=100, alpha=0.7, c='blue')
    ax1.set_xlabel('参数量 (M)', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('准确率 vs 参数量', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加模型名称标签
    for i, name in enumerate(model_names):
        ax1.annotate(name.replace('_', '\n'), (parameters[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 准确率 vs 训练时间散点图
    scatter2 = ax2.scatter(train_times, accuracies, s=100, alpha=0.7, c='red')
    ax2.set_xlabel('训练时间 (s)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('准确率 vs 训练时间', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加模型名称标签
    for i, name in enumerate(model_names):
        ax2.annotate(name.replace('_', '\n'), (train_times[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('assets/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(results, model_names=None):
    """绘制训练曲线"""
    if model_names is None:
        # 选择几个代表性模型
        model_names = ['resnet_20', 'eca_resnet_20', 'ghost_resnet_20', 'convnext_tiny_timm']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    for i, model_name in enumerate(model_names[:4]):
        if model_name in results and 'epochs' in results[model_name]:
            result = results[model_name]
            epochs = result['epochs']
            train_acc = result['train_acc']
            test_acc = result['test_acc']
            
            ax = axes[i]
            ax.plot(epochs, train_acc, 'b-', label='训练准确率', linewidth=2)
            ax.plot(epochs, test_acc, 'r-', label='测试准确率', linewidth=2)
            ax.set_title(f'{model_name} 训练曲线', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('准确率 (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标注最佳准确率
            best_epoch = epochs[test_acc.index(max(test_acc))]
            best_acc = max(test_acc)
            ax.annotate(f'最佳: {best_acc:.1f}%', 
                       xy=(best_epoch, best_acc), xytext=(10, 10),
                       textcoords='offset points', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('assets/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_report(results):
    """创建综合报告"""
    print("=" * 80)
    print("CIFAR-100 分类任务 - 十种先进架构对比分析报告")
    print("=" * 80)
    
    # 基本统计
    valid_results = {k: v for k, v in results.items() if 'error' not in v and 'best_acc' in v}
    print(f"\n总模型数量: {len(results)}")
    print(f"成功训练: {len(valid_results)}")
    print(f"失败模型: {len(results) - len(valid_results)}")
    
    if not valid_results:
        print("没有可用的训练结果!")
        return
    
    # 性能排名
    print(f"\n🏆 性能排名 (Top-1准确率):")
    print("-" * 60)
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_models[:10], 1):
        print(f"{i:2d}. {model_name:<20} {result['best_acc']:6.2f}% "
              f"(Top5: {result.get('best_acc_top5', 0):5.1f}%, "
              f"参数: {result['parameters']:5.2f}M)")
    
    # 效率分析
    print(f"\n⚡ 效率分析:")
    print("-" * 60)
    
    # 参数效率排名
    param_efficiency = [(name, res['best_acc'] / res['parameters'] if res['parameters'] > 0 else 0) 
                       for name, res in valid_results.items()]
    param_efficiency.sort(key=lambda x: x[1], reverse=True)
    
    print("参数效率排名 (准确率/参数量):")
    for i, (name, eff) in enumerate(param_efficiency[:5], 1):
        acc = valid_results[name]['best_acc']
        params = valid_results[name]['parameters']
        print(f"  {i}. {name:<20} {eff:6.2f} ({acc:.1f}% / {params:.2f}M)")
    
    # 速度分析
    speed_ranking = sorted(valid_results.items(), key=lambda x: x[1]['total_time'])
    print(f"\n训练速度排名:")
    for i, (name, res) in enumerate(speed_ranking[:5], 1):
        print(f"  {i}. {name:<20} {res['total_time']:6.1f}s "
              f"(准确率: {res['best_acc']:.1f}%)")
    
    # 技术特点分析
    print(f"\n🔬 技术特点分析:")
    print("-" * 60)
    
    categories = {
        '基础网络': ['resnet_20', 'resnet_32', 'resnet_56'],
        '注意力机制': ['eca_resnet_20', 'eca_resnet_32'],
        '轻量化设计': ['ghost_resnet_20', 'ghost_resnet_32', 'ghostnet_100'],
        '现代化架构': ['convnext_tiny', 'convnext_tiny_timm'],
        '多尺度感知': ['segnext_mscan_tiny'],
        '混合架构': ['coatnet_0'],
        '跨阶段网络': ['cspresnet50'],
        '分裂注意力': ['resnest50d'],
        'MLP架构': ['mlp_mixer_tiny', 'mlp_mixer_b16'],
        '替代架构': ['hornet_tiny']
    }
    
    for category, models in categories.items():
        category_results = [(name, valid_results[name]) for name in models if name in valid_results]
        if category_results:
            avg_acc = np.mean([res['best_acc'] for _, res in category_results])
            avg_params = np.mean([res['parameters'] for _, res in category_results])
            print(f"{category:<12}: 平均准确率 {avg_acc:5.1f}%, 平均参数量 {avg_params:5.2f}M")

def main():
    """主函数"""
    # 创建输出目录
    Path('assets').mkdir(exist_ok=True)
    
    # 加载结果
    print("加载训练结果...")
    results = load_results()
    
    if not results:
        print("没有找到训练结果文件!")
        return
    
    # 创建汇总表
    print("创建结果汇总表...")
    summary_df = create_summary_table(results)
    summary_df.to_csv('assets/model_comparison_summary.csv', index=False, encoding='utf-8')
    print(f"汇总表已保存到: assets/model_comparison_summary.csv")
    
    # 生成可视化图表
    print("生成可视化图表...")
    plot_accuracy_comparison(results)
    plot_efficiency_analysis(results)
    plot_training_curves(results)
    
    # 生成综合报告
    create_comprehensive_report(results)
    
    print(f"\n📊 分析完成! 图表已保存到 assets/ 目录")
    print("生成的文件:")
    print("  - accuracy_comparison.png: 准确率对比图")
    print("  - efficiency_analysis.png: 效率分析图") 
    print("  - training_curves.png: 训练曲线图")
    print("  - model_comparison_summary.csv: 结果汇总表")

if __name__ == "__main__":
    main() 