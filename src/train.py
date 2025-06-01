import torch
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import time

from models import (
    resnet20,
    resnet32,
    resnet56,
    resnet20_slim,
    resnet32_slim,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    count_parameters,
)
from dataset import get_dataloaders
from trainer import Trainer


def print_model_summary(model, model_name):
    """打印模型详细信息"""
    print(f"\n{'='*60}")
    print(f"Model Architecture: {model_name}")
    print(f"{'='*60}")

    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:.2f}M")
    print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
    print(f"Model Size (approx): {total_params * 4:.2f} MB")  # 假设 float32

    # 打印模型结构概览
    print(f"\nModel Structure:")
    print(f"├── Input: 3 channels (RGB)")
    
    if "convnext" in model_name:
        # ConvNeXt 架构信息
        if "tiny" in model_name:
            depths, dims = [2, 2, 6, 2], [48, 96, 192, 384]
        elif "small" in model_name:
            depths, dims = [2, 2, 18, 2], [48, 96, 192, 384]
        elif "base" in model_name:
            depths, dims = [2, 2, 18, 2], [64, 128, 256, 512]
        elif "large" in model_name:
            depths, dims = [2, 2, 18, 2], [96, 192, 384, 768]
        else:
            depths, dims = [2, 2, 6, 2], [48, 96, 192, 384]
        
        print(f"├── Stem: 4×4 conv, stride=4 → {dims[0]} channels")
        print(f"├── Stage1: {depths[0]} blocks, {dims[0]} channels")
        print(f"├── Stage2: {depths[1]} blocks, {dims[1]} channels") 
        print(f"├── Stage3: {depths[2]} blocks, {dims[2]} channels")
        print(f"├── Stage4: {depths[3]} blocks, {dims[3]} channels")
        print(f"├── Features: 7×7 DWConv + LayerNorm + InvertedBottleneck")
        print(f"├── GlobalAvgPool: → {dims[3]} features")
        print(f"└── FC: {dims[3]} → 100 classes")
    else:
        # ResNet 架构信息
        print(f"├── Conv1: 3×3, stride=1, padding=1")
        
        if "resnet20" in model_name:
            layers = [3, 3, 3]
        elif "resnet32" in model_name:
            layers = [5, 5, 5]
        elif "resnet56" in model_name:
            layers = [9, 9, 9]
        else:
            layers = [3, 3, 3]

        width_mult = 0.5 if "slim" in model_name else 1.0
        channels = [int(16 * width_mult), int(32 * width_mult), int(64 * width_mult)]

        print(f"├── Layer1: {layers[0]} blocks, {channels[0]} channels")
        print(f"├── Layer2: {layers[1]} blocks, {channels[1]} channels")
        print(f"├── Layer3: {layers[2]} blocks, {channels[2]} channels")
        print(f"├── AvgPool: Adaptive (1×1)")
        print(f"└── FC: {channels[2]} → 100 classes")

    print(f"{'='*60}\n")


def print_training_config(args):
    """打印训练配置"""
    print(f"Training Configuration:")
    print(f"├── Epochs: {args.epochs}")
    print(f"├── Batch Size: {args.batch_size}")
    print(f"├── Learning Rate: {args.lr}")
    print(f"├── Weight Decay: {args.weight_decay}")
    print(f"├── Data Augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    print(f"├── Data Directory: {args.data_dir}")
    print(f"└── Save Directory: {args.save_dir}")
    print()


def print_dataset_info(train_loader, test_loader):
    """打印数据集信息"""
    print(f"Dataset Information:")
    print(f"├── Dataset: CIFAR-100")
    print(f"├── Classes: 100")
    print(f"├── Train Samples: {len(train_loader.dataset):,}")
    print(f"├── Test Samples: {len(test_loader.dataset):,}")
    print(f"├── Train Batches: {len(train_loader)}")
    print(f"├── Test Batches: {len(test_loader)}")
    print(f"└── Image Size: 32×32×3")
    print()


def print_system_info():
    """打印系统信息"""
    print(f"System Information:")
    print(f"├── Python Version: {sys.version.split()[0]}")
    print(f"├── PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"├── CUDA Available: Yes")
        print(f"├── CUDA Version: {torch.version.cuda}")
        print(f"├── GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"├── GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        print(f"└── Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"├── CUDA Available: No")
        print(f"└── Device: CPU")
    print()


def test_model_forward(model, device, batch_size=2):
    """测试模型前向传播"""
    print(f"Testing model forward pass...")

    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()

    # 创建测试输入
    test_input = torch.randn(batch_size, 3, 32, 32).to(device)

    try:
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        forward_time = time.time() - start_time

        print(f"✓ Forward pass successful")
        print(f"├── Input shape: {list(test_input.shape)}")
        print(f"├── Output shape: {list(output.shape)}")
        print(f"├── Forward time: {forward_time*1000:.2f}ms")
        print(f"└── Device: {next(model.parameters()).device}")
        print()
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input device: {test_input.device}")
        return False


def get_default_lr_and_optimizer(model_name):
    """根据模型类型返回默认学习率和优化器设置"""
    if "convnext" in model_name:
        # ConvNeXt 论文建议使用 AdamW 和较小的学习率
        return 0.004, "adamw"
    else:
        # ResNet 使用 SGD 和较大的学习率
        return 0.1, "sgd"


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Framework for CIFAR-100")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        choices=[
            "resnet20", "resnet32", "resnet56", "resnet20_slim", "resnet32_slim",
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        ],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate (auto-selected if not specified)")
    parser.add_argument(
        "--optimizer", type=str, default=None, 
        choices=["sgd", "adamw"],
        help="Optimizer type (auto-selected if not specified)"
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

    # 自动选择学习率和优化器
    if args.lr is None or args.optimizer is None:
        default_lr, default_optimizer = get_default_lr_and_optimizer(args.model)
        if args.lr is None:
            args.lr = default_lr
        if args.optimizer is None:
            args.optimizer = default_optimizer
        print(f"Auto-selected: LR={args.lr}, Optimizer={args.optimizer}")

    # 打印启动横幅
    print(f"\n{'='*80}")
    print(f"{'Deep Learning Training Framework':^80}")
    print(f"{'='*80}")

    # 系统信息
    print_system_info()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型创建
    model_dict = {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet56": resnet56,
        "resnet20_slim": resnet20_slim,
        "resnet32_slim": resnet32_slim,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
    }

    print(f"Creating model: {args.model}")
    model = model_dict[args.model]()

    # 打印模型详细信息
    print_model_summary(model, args.model)

    # 测试模型前向传播
    if not test_model_forward(model, device, args.batch_size):
        print("Model forward pass test failed. Exiting...")
        return

    # 数据加载
    print(f"Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment=not args.no_augment
    )

    # 打印数据集信息
    print_dataset_info(train_loader, test_loader)

    # 打印训练配置
    print_training_config(args)

    # 创建保存目录
    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # 估算训练时间
    estimated_time_per_epoch = (
        len(train_loader) * args.batch_size / 50000 * 60
    )  # 粗略估算
    estimated_total_time = estimated_time_per_epoch * args.epochs / 60
    print(f"Estimated training time: ~{estimated_total_time:.1f} minutes")

    # 开始训练
    print(f"\n{'='*80}")
    print(f"{'Starting Training':^80}")
    print(f"{'='*80}")

    trainer = Trainer(model, device, save_dir, verbose=args.verbose)
    history = trainer.fit(
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
    )

    # 训练完成
    print(f"\n{'='*80}")
    print(f"{'Training Completed':^80}")
    print(f"{'='*80}")

    # 生成训练曲线
    print(f"Generating training curves...")
    trainer.plot_curves(save_dir / "training_curves.png")

    # 最终总结
    print(f"\nTraining Summary:")
    print(f"├── Model: {args.model}")
    print(f"├── Best Test Accuracy: {trainer.best_acc:.2f}%")
    print(f"├── Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"├── Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"├── Checkpoints saved to: {save_dir}")
    print(f"└── Training curves saved to: {save_dir / 'training_curves.png'}")

    print(f"\n{'='*80}")
    print(f"{'All Done!':^80}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()