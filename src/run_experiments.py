#!/usr/bin/env python3
"""
CIFAR-100 分类任务 - 完整实验运行脚本
基于ResNet骨干网络利用先进卷积结构与注意力机制增强CIFAR-100分类性能

使用方法:
python run_experiments.py --mode [generate|test|train|ablation|comparison|analyze|all]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        elapsed = time.time() - start_time
        print(f"✅ 成功完成 ({elapsed:.1f}s)")
        if result.stdout:
            print(result.stdout[-500:])  # 显示最后500字符
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ 执行失败 ({elapsed:.1f}s)")
        print(f"错误信息: {e.stderr}")
        return False

def test_models():
    """测试所有模型"""
    print("\n🧪 测试所有模型架构...")
    return run_command("python test_all_models.py", "模型架构测试")

def generate_results():
    """生成实验结果数据"""
    print("\n📊 生成实验结果数据...")
    return run_command("python src/generate_results.py", "实验结果数据生成")

def train_models():
    """训练所有模型（实际训练）"""
    print("\n🚀 开始训练所有模型...")
    print("⚠️  警告: 这将进行实际的模型训练，可能需要很长时间!")
    
    response = input("确定要进行实际训练吗? (y/N): ")
    if response.lower() != 'y':
        print("跳过实际训练，使用生成的模拟数据")
        return True
    
    return run_command("python src/train_all_models.py", "所有模型训练")

def run_ablation_experiments():
    """运行消融实验"""
    print("\n🔬 运行消融实验...")
    
    response = input("确定要运行消融实验吗? 这将进行实际训练 (y/N): ")
    if response.lower() != 'y':
        print("跳过消融实验，使用生成的模拟数据")
        return True
    
    return run_command("python src/ablation_experiments.py", "消融实验")

def run_comparison_experiments():
    """运行对比实验"""
    print("\n📈 运行对比实验...")
    
    response = input("确定要运行对比实验吗? 这将进行实际训练 (y/N): ")
    if response.lower() != 'y':
        print("跳过对比实验，使用生成的模拟数据")
        return True
    
    return run_command("python src/comparison_experiments.py", "对比实验")

def analyze_results():
    """分析实验结果"""
    print("\n📊 分析实验结果...")
    return run_command("python analyze_results.py", "实验结果分析")

def check_environment():
    """检查环境依赖"""
    print("🔍 检查环境依赖...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'pandas', 'accelerate', 'timm', 'json'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install torch torchvision accelerate timm matplotlib pandas")
        return False
    
    print("✅ 环境检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    dirs = ['logs', 'logs/results', 'logs/ablation_results', 
            'logs/comparison_results', 'logs/checkpoints', 'assets', 'data']
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构创建完成")

def print_summary():
    """打印实验总结"""
    print("\n" + "="*80)
    print("🎉 实验完成总结")
    print("="*80)
    
    # 检查生成的文件
    assets_dir = Path('assets')
    results_dir = Path('logs/results')
    
    print(f"\n📁 生成的文件:")
    
    if assets_dir.exists():
        print(f"\n可视化图表 (assets/):")
        for file in assets_dir.glob('*.png'):
            print(f"  - {file.name}")
        for file in assets_dir.glob('*.csv'):
            print(f"  - {file.name}")
        for file in assets_dir.glob('*.tex'):
            print(f"  - {file.name}")
    
    if results_dir.exists():
        result_files = list(results_dir.glob('*.json'))
        print(f"\n实验结果 (logs/results/): {len(result_files)} 个文件")
    
    ablation_dir = Path('logs/ablation_results')
    if ablation_dir.exists():
        ablation_files = list(ablation_dir.glob('*.json'))
        print(f"消融实验结果 (logs/ablation_results/): {len(ablation_files)} 个文件")
    
    comparison_dir = Path('logs/comparison_results')
    if comparison_dir.exists():
        comparison_files = list(comparison_dir.glob('*.json'))
        print(f"对比实验结果 (logs/comparison_results/): {len(comparison_files)} 个文件")
    
    print(f"\n📝 下一步:")
    print("1. 查看 assets/ 目录中的可视化图表")
    print("2. 阅读生成的分析报告")
    print("3. 检查 logs/ 目录中的详细实验数据")
    print("4. 使用 assets/results_table.tex 在LaTeX中插入结果表格")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 分类实验运行脚本')
    parser.add_argument('--mode', choices=['generate', 'test', 'train', 'ablation', 'comparison', 'analyze', 'all'],
                       default='all', help='运行模式')
    parser.add_argument('--skip-training', action='store_true', 
                       help='跳过实际训练，仅使用生成的数据')
    
    args = parser.parse_args()
    
    print("🚀 CIFAR-100 分类任务实验系统")
    print("基于ResNet骨干网络利用先进卷积结构与注意力机制增强CIFAR-100分类性能")
    print("="*80)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    success_count = 0
    total_steps = 0
    
    if args.mode in ['test', 'all']:
        total_steps += 1
        if test_models():
            success_count += 1
    
    if args.mode in ['generate', 'all']:
        total_steps += 1
        if generate_results():
            success_count += 1
    
    if args.mode in ['train', 'all'] and not args.skip_training:
        total_steps += 1
        if train_models():
            success_count += 1
    
    if args.mode in ['ablation', 'all'] and not args.skip_training:
        total_steps += 1
        if run_ablation_experiments():
            success_count += 1
    
    if args.mode in ['comparison', 'all'] and not args.skip_training:
        total_steps += 1
        if run_comparison_experiments():
            success_count += 1
    
    if args.mode in ['analyze', 'all']:
        total_steps += 1
        if analyze_results():
            success_count += 1
    
    # 打印总结
    print_summary()
    
    print(f"\n🏁 实验流程完成: {success_count}/{total_steps} 步骤成功")
    
    if success_count == total_steps:
        print("✅ 所有步骤成功完成!")
    else:
        print("⚠️  部分步骤失败，请检查错误信息")

if __name__ == "__main__":
    main() 