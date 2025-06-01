import torch
import argparse
from pathlib import Path
import sys
import os
from tqdm import tqdm
import time
import logging
import json
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

# 尝试导入模型架构打印工具
try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

try:
    from torchsummary import summary as torchsummary_summary
    HAS_TORCHSUMMARY = True
except ImportError:
    HAS_TORCHSUMMARY = False

from models import (
    resnet20,
    resnet32,
    resnet56,
    resnet20_slim,
    resnet32_slim,
    improved_resnet20_v1,
    improved_resnet20_v2,
    improved_resnet20_v3,
    improved_resnet20_v4,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    count_parameters,
)
from dataset import get_dataloaders
from trainer import Trainer


class ModelLogger:
    """独立的模型日志记录器"""
    
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.log_dir = save_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{model_name}_{timestamp}.log"
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"{model_name}_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 控制台输出缓冲
        self.console_buffer = []
    
    def info(self, message):
        """记录信息，处理Unicode字符"""
        # 替换问题字符
        safe_message = self._safe_unicode(message)
        
        # 写入日志文件
        self.logger.info(safe_message)
        
        # 添加到控制台缓冲
        self.console_buffer.append(safe_message)
    
    def error(self, message):
        """记录错误"""
        safe_message = self._safe_unicode(message)
        self.logger.error(safe_message)
        self.console_buffer.append(f"ERROR: {safe_message}")
    
    def _safe_unicode(self, message):
        """处理Unicode字符"""
        replacements = {
            '✓': '[OK]',
            '✗': '[FAIL]', 
            '🎉': '[BEST]',
            '├': '|-',
            '└': '`-',
            '→': '->',
            '×': 'x'
        }
        
        for old, new in replacements.items():
            message = message.replace(old, new)
        
        return message
    
    def flush_console(self):
        """刷新控制台输出"""
        if self.console_buffer:
            # 清屏并显示当前模型的日志
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{'='*80}")
            print(f"Training Model: {self.model_name}")
            print(f"{'='*80}")
            
            # 显示最近的日志
            for line in self.console_buffer[-50:]:  # 只显示最近50行
                print(line)
            
            print(f"{'='*80}")
    
    def clear_buffer(self):
        """清空控制台缓冲"""
        self.console_buffer = []


def get_model_architecture(model, model_name, input_size=(1, 3, 32, 32)):
    """获取模型架构信息"""
    arch_info = []
    
    # 基本信息
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    arch_info.extend([
        f"Model: {model_name}",
        f"Total Parameters: {total_params:.2f}M",
        f"Trainable Parameters: {trainable_params/1e6:.2f}M",
        f"Model Size (approx): {total_params * 4:.2f} MB",
        ""
    ])
    
    # 尝试使用 torchinfo - 直接使用传入的模型实例
    if HAS_TORCHINFO:
        try:
            arch_info.append("Model Architecture (torchinfo):")
            arch_info.append("-" * 60)
            
            # 确保模型在CPU上进行summary（避免设备问题）
            model_for_summary = model.cpu()
            model_for_summary.eval()
            
            # 使用 torchinfo 获取详细信息
            summary_str = str(torchinfo_summary(
                model_for_summary, 
                input_size=input_size,
                verbose=0,
                col_names=["input_size", "output_size", "num_params", "kernel_size"],
                row_settings=["var_names"],
                device='cpu'
            ))
            
            # 分行添加
            for line in summary_str.split('\n'):
                if line.strip():  # 跳过空行
                    arch_info.append(line)
            
            arch_info.append("-" * 60)
            
        except Exception as e:
            arch_info.extend([
                f"torchinfo failed: {str(e)}",
                "Falling back to torchsummary or basic info..."
            ])
    
    # 如果 torchinfo 失败，尝试 torchsummary
    if (not HAS_TORCHINFO or "failed" in arch_info[-2]) and HAS_TORCHSUMMARY:
        try:
            arch_info.append("Model Architecture (torchsummary):")
            arch_info.append("-" * 60)
            
            # 确保模型在CPU上
            model_for_summary = model.cpu()
            model_for_summary.eval()
            
            # 创建字符串缓冲区
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            # 调用 torchsummary
            torchsummary_summary(model_for_summary, input_size[1:])  # 不包括batch维度
            
            # 恢复stdout并获取输出
            sys.stdout = old_stdout
            summary_output = buffer.getvalue()
            
            # 添加输出到架构信息
            for line in summary_output.split('\n'):
                if line.strip():
                    arch_info.append(line)
            
            arch_info.append("-" * 60)
            
        except Exception as e:
            arch_info.extend([
                f"torchsummary failed: {str(e)}",
                "Using basic architecture info..."
            ])
        finally:
            sys.stdout = old_stdout
    
    # 如果都失败了，或者都没有，使用基本的模型字符串表示
    if (not HAS_TORCHINFO and not HAS_TORCHSUMMARY) or "failed" in str(arch_info):
        arch_info.extend([
            "",
            "Model Architecture (basic representation):",
            "-" * 60
        ])
        
        # 获取模型的字符串表示
        model_str = str(model)
        lines = model_str.split('\n')
        
        # 显示模型结构，但限制行数
        for i, line in enumerate(lines):
            if i > 30:  # 限制显示行数
                arch_info.append(f"... (truncated, {len(lines) - i} more lines)")
                break
            arch_info.append(line)
        
        arch_info.extend([
            "",
            "For detailed architecture info, install:",
            "  pip install torchinfo  (recommended)",
            "  pip install torchsummary",
            "-" * 60
        ])
    
    return arch_info


def get_simple_model_info(model, model_name):
    """获取简化的模型信息（用于快速显示）"""
    total_params = count_parameters(model)
    
    # 根据模型类型提供简要描述
    if "resnet" in model_name and "improved" not in model_name:
        if "20" in model_name:
            desc = "ResNet-20: 3 layers x 3 blocks each"
        elif "32" in model_name:
            desc = "ResNet-32: 3 layers x 5 blocks each"
        elif "56" in model_name:
            desc = "ResNet-56: 3 layers x 9 blocks each"
        else:
            desc = "ResNet variant"
            
        if "slim" in model_name:
            desc += " (0.5x channels)"
    
    elif "improved" in model_name:
        base_desc = "Enhanced ResNet-20"
        if "v1" in model_name:
            desc = f"{base_desc} v1: 7x7 kernels for larger receptive field"
        elif "v2" in model_name:
            desc = f"{base_desc} v2: Depthwise conv + Inverted bottleneck"
        elif "v3" in model_name:
            desc = f"{base_desc} v3: LayerNorm + GELU + Layer Scale + Drop Path"
        elif "v4" in model_name:
            desc = f"{base_desc} v4: ConvNeXt-inspired design"
        else:
            desc = f"{base_desc}: Advanced improvements"
    
    elif "convnext" in model_name:
        if "tiny" in model_name:
            desc = "ConvNeXt-Tiny: [2,2,6,2] blocks, [48,96,192,384] dims"
        elif "small" in model_name:
            desc = "ConvNeXt-Small: [2,2,18,2] blocks, [48,96,192,384] dims"
        elif "base" in model_name:
            desc = "ConvNeXt-Base: [2,2,18,2] blocks, [64,128,256,512] dims"
        elif "large" in model_name:
            desc = "ConvNeXt-Large: [2,2,18,2] blocks, [96,192,384,768] dims"
        else:
            desc = "ConvNeXt variant: Modern CNN architecture"
    else:
        desc = "Custom model architecture"
    
    return [
        f"Model: {model_name}",
        f"Description: {desc}",
        f"Parameters: {total_params:.2f}M ({total_params*1e6:,.0f} total)",
        f"Model size: ~{total_params * 4:.1f} MB (FP32)",
        f"Input: 32x32x3 RGB -> Output: 100 classes (CIFAR-100)"
    ]


def print_layer_summary(model, model_name):
    """手动打印模型层的简要总结"""
    summary_lines = [
        "",
        f"Layer Summary for {model_name}:",
        "-" * 50
    ]
    
    total_params = 0
    layer_count = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_count += 1
                total_params += params
                
                # 获取层类型
                layer_type = type(module).__name__
                
                # 获取层的基本信息
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    info = f"{module.in_features} -> {module.out_features}"
                elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    kernel = getattr(module, 'kernel_size', 'N/A')
                    stride = getattr(module, 'stride', 'N/A')
                    info = f"{module.in_channels} -> {module.out_channels}, kernel={kernel}, stride={stride}"
                elif hasattr(module, 'num_features'):
                    info = f"features={module.num_features}"
                else:
                    info = "N/A"
                
                summary_lines.append(f"{layer_count:2d}. {name:<25} {layer_type:<15} {info:<25} {params:>8,} params")
    
    summary_lines.extend([
        "-" * 50,
        f"Total: {layer_count} layers, {total_params:,} parameters",
        ""
    ])
    
    return summary_lines


def get_default_lr_and_optimizer(model_name):
    """根据模型类型返回默认学习率和优化器设置"""
    if "convnext" in model_name:
        return 0.004, "adamw"
    elif "improved" in model_name and ("v3" in model_name or "v4" in model_name):
        return 0.001, "adamw"
    else:
        return 0.1, "sgd"


def save_experiment_config(save_dir, args, model_params, model_name):
    """保存实验配置"""
    config = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "data_augmentation": not args.no_augment,
        "model_parameters": model_params,
        "timestamp": datetime.now().isoformat()
    }
    
    config_file = save_dir / "experiment_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config_file


def test_model_forward(model, device, batch_size=2, logger=None):
    """测试模型前向传播"""
    logger.info("Testing model forward pass...")

    model = model.to(device)
    model.eval()
    test_input = torch.randn(batch_size, 3, 32, 32).to(device)

    try:
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        forward_time = time.time() - start_time

        logger.info(f"[OK] Forward pass successful")
        logger.info(f"|- Input shape: {list(test_input.shape)}")
        logger.info(f"|- Output shape: {list(output.shape)}")
        logger.info(f"|- Forward time: {forward_time*1000:.2f}ms")
        logger.info(f"`- Device: {next(model.parameters()).device}")
        
        return True
    except Exception as e:
        logger.error(f"[FAIL] Forward pass failed: {e}")
        logger.error(f"Model device: {next(model.parameters()).device}")
        logger.error(f"Input device: {test_input.device}")
        return False


def train_single_model(model_name, args):
    """训练单个模型"""
    # 创建模型专用日志记录器
    save_dir = Path(args.save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ModelLogger(Path(args.save_dir), model_name)
    
    logger.info("="*80)
    logger.info(f"Starting training for model: {model_name}")
    logger.info("="*80)
    
    # 模型创建
    model_dict = {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet56": resnet56,
        "resnet20_slim": resnet20_slim,
        "resnet32_slim": resnet32_slim,
        "improved_resnet20_v1": improved_resnet20_v1,
        "improved_resnet20_v2": improved_resnet20_v2,
        "improved_resnet20_v3": improved_resnet20_v3,
        "improved_resnet20_v4": improved_resnet20_v4,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Creating model: {model_name}")
    model = model_dict[model_name]()
    
    # 首先显示简要信息
    simple_info = get_simple_model_info(model, model_name)
    for line in simple_info:
        logger.info(line)
    
    # 手动打印层总结
    layer_summary = print_layer_summary(model, model_name)
    for line in layer_summary:
        logger.info(line)
    
    # 尝试获取详细的模型架构信息
    if HAS_TORCHINFO or HAS_TORCHSUMMARY:
        logger.info("Getting detailed architecture...")
        try:
            arch_info = get_model_architecture(model, model_name)
            for line in arch_info:
                logger.info(line)
        except Exception as e:
            logger.error(f"Failed to get detailed architecture: {e}")
            logger.info("Continuing with basic model info...")
    
    # 获取优化器设置
    if args.lr is None or args.optimizer is None:
        default_lr, default_optimizer = get_default_lr_and_optimizer(model_name)
        lr = args.lr if args.lr is not None else default_lr
        optimizer_type = args.optimizer if args.optimizer is not None else default_optimizer
    else:
        lr = args.lr
        optimizer_type = args.optimizer
    
    logger.info("")
    logger.info(f"Training Configuration:")
    logger.info(f"|- Learning Rate: {lr}")
    logger.info(f"|- Optimizer: {optimizer_type}")
    logger.info(f"|- Weight Decay: {args.weight_decay}")
    logger.info(f"|- Batch Size: {args.batch_size}")
    logger.info(f"|- Epochs: {args.epochs}")

    # 测试模型前向传播
    if not test_model_forward(model, device, args.batch_size, logger):
        logger.error("Model forward pass test failed. Skipping...")
        return None

    # 数据加载
    logger.info("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment=not args.no_augment
    )

    # 保存实验配置
    config_file = save_experiment_config(save_dir, args, count_parameters(model), model_name)
    logger.info(f"Experiment config saved to: {config_file}")

    # 刷新控制台显示
    logger.flush_console()

    # 开始训练
    logger.info(f"Starting training for {args.epochs} epochs...")
    trainer = Trainer(model, device, save_dir, verbose=args.verbose)
    
    try:
        history = trainer.fit(
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=lr,
            optimizer_type=optimizer_type,
            weight_decay=args.weight_decay,
        )
        
        # 训练完成后的处理
        logger.info("Training completed successfully!")
        logger.info(f"Best test accuracy: {trainer.best_acc:.2f}%")
        
        # 生成训练曲线
        trainer.plot_curves(save_dir / "training_curves.png")
        
        # 导出模型
        onnx_path = trainer.export_model(model_name=model_name, input_size=(1, 3, 32, 32))
        script_path = trainer.export_torchscript(model_name=model_name, input_size=(1, 3, 32, 32))
        summary_path = trainer.save_model_summary(model_name=model_name)
        
        # 保存训练结果
        results = {
            "model": model_name,
            "best_accuracy": trainer.best_acc,
            "final_train_accuracy": history['train_acc'][-1],
            "final_test_accuracy": history['test_acc'][-1],
            "total_epochs": args.epochs,
            "parameters": count_parameters(model),
            "training_time": time.time() - trainer.start_time if hasattr(trainer, 'start_time') else None
        }
        
        results_file = save_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training results saved to: {results_file}")
        
        # 最终刷新显示
        logger.flush_console()
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for {model_name}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning Training Framework for CIFAR-100"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        help="Model architecture to train (or 'all' for batch training)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Multiple models to train (alternative to --model)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate (auto-selected if not specified)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adamw"],
        help="Optimizer type (auto-selected if not specified)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay coefficient"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # 检查依赖
    if not HAS_TORCHINFO and not HAS_TORCHSUMMARY:
        print("Warning: Neither torchinfo nor torchsummary is installed.")
        print("Install one for better model architecture display:")
        print("  pip install torchinfo  (recommended)")
        print("  pip install torchsummary")
        print()

    # 确定要训练的模型列表
    all_models = [
        "resnet20",
        "improved_resnet20_v1", "improved_resnet20_v2", "improved_resnet20_v3", "improved_resnet20_v4",
        "convnext_tiny",
    ]
    
    if args.models:
        models_to_train = args.models
    elif args.model == "all":
        models_to_train = ["resnet20", "convnext_tiny", "improved_resnet20_v1", "improved_resnet20_v2", "improved_resnet20_v3", "improved_resnet20_v4"]
    elif args.model == "resnet_all":
        models_to_train = ["resnet20", "resnet32", "resnet56", "resnet20_slim", "resnet32_slim"]
    elif args.model == "improved_all":
        models_to_train = ["improved_resnet20_v1", "improved_resnet20_v2", "improved_resnet20_v3", "improved_resnet20_v4"]
    elif args.model == "convnext_all":
        models_to_train = ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
    else:
        models_to_train = [args.model]

    # 验证模型名称
    available_models = list(set(all_models + ["all", "resnet_all", "improved_all", "convnext_all"]))
    invalid_models = [m for m in models_to_train if m not in all_models]
    if invalid_models:
        print(f"Error: Invalid model names: {invalid_models}")
        print(f"Available models: {available_models}")
        return

    # 创建主日志目录
    main_save_dir = Path(args.save_dir)
    main_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"{'Batch Training Started':^80}")
    print(f"{'='*80}")
    print(f"Models to train: {models_to_train}")
    print(f"Total models: {len(models_to_train)}")
    print(f"Architecture tool: {'torchinfo' if HAS_TORCHINFO else 'torchsummary' if HAS_TORCHSUMMARY else 'basic'}")
    
    # 批量训练
    all_results = []
    successful_trains = 0
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n{'='*80}")
        print(f"Training {i}/{len(models_to_train)}: {model_name}")
        print(f"{'='*80}")
        
        result = train_single_model(model_name, args)
        
        if result:
            all_results.append(result)
            successful_trains += 1
            print(f"[OK] {model_name} training completed successfully!")
        else:
            print(f"[FAIL] {model_name} training failed!")
    
    # 保存批量训练总结
    summary = {
        "total_models": len(models_to_train),
        "successful_trains": successful_trains,
        "failed_trains": len(models_to_train) - successful_trains,
        "models_trained": models_to_train,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = main_save_dir / "batch_training_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 清屏并显示最终总结
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{'='*80}")
    print(f"{'Batch Training Summary':^80}")
    print(f"{'='*80}")
    print(f"Total models: {len(models_to_train)}")
    print(f"Successful: {successful_trains}")
    print(f"Failed: {len(models_to_train) - successful_trains}")
    
    if all_results:
        print(f"\nResults Summary:")
        for result in sorted(all_results, key=lambda x: x['best_accuracy'], reverse=True):
            print(f"|- {result['model']}: {result['best_accuracy']:.2f}%")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual logs saved in: {main_save_dir / 'logs'}")
    if not HAS_TORCHINFO:
        print(f"Tip: Install torchinfo for better model display: pip install torchinfo")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()