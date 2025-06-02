#!/bin/bash

# CoAtNet-CIFAROpt 训练脚本
# DL-2025项目 - 基于CoAtNet的CIFAR-100图像分类创新方案

echo "=========================================="
echo "开始CoAtNet-CIFAROpt训练实验"
echo "DL-2025项目 - CIFAR-100图像分类"
echo "=========================================="

# 激活conda环境
source /root/data-tmp/miniconda3/etc/profile.d/conda.sh
conda activate llm

# 检查Python环境
echo "Python版本:"
python --version
echo ""

echo "PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
echo ""

# 创建必要的目录
mkdir -p data
mkdir -p logs
mkdir -p src

# 进入源代码目录
cd src

echo "开始训练CoAtNet-CIFAROpt标准版本..."
echo "=========================================="

# 训练CoAtNet-CIFAROpt标准版本
python train_coatnet_cifar_opt.py \
    --model_name "coatnet_cifar_opt" \
    --num_epochs 300 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --weight_decay 0.02 \
    --warmup_epochs 10 \
    --label_smoothing 0.1 \
    --mixup_alpha 1.0 \
    --cutmix_alpha 1.0 \
    --stochastic_depth_rate 0.1 \
    --save_interval 50

echo ""
echo "CoAtNet-CIFAROpt标准版本训练完成!"
echo ""

# 等待用户确认是否继续训练大核版本
read -p "是否继续训练CoAtNet-CIFAROpt大核版本? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始训练CoAtNet-CIFAROpt大核版本..."
    echo "=========================================="
    
    # 训练CoAtNet-CIFAROpt大核版本
    python train_coatnet_cifar_opt.py \
        --model_name "coatnet_cifar_opt_large_stem" \
        --num_epochs 300 \
        --batch_size 128 \
        --learning_rate 5e-4 \
        --weight_decay 0.02 \
        --warmup_epochs 10 \
        --label_smoothing 0.1 \
        --mixup_alpha 1.0 \
        --cutmix_alpha 1.0 \
        --stochastic_depth_rate 0.1 \
        --save_interval 50
    
    echo ""
    echo "CoAtNet-CIFAROpt大核版本训练完成!"
fi

echo ""
echo "=========================================="
echo "训练实验完成！"
echo "查看logs目录中的训练结果和模型权重"
echo "=========================================="

# 显示训练结果总结
echo "训练结果总结:"
echo "----------------------------------------"
find ../logs -name "best_model.pth" -exec dirname {} \; | while read dir; do
    if [ -f "$dir/training_history.npz" ]; then
        echo "实验目录: $(basename $dir)"
        python -c "
import numpy as np
import os
import sys
sys.path.append('.')

# 读取训练历史
data = np.load('$dir/training_history.npz')
best_top1 = max(data['test_top1_accs'])
best_top5 = max(data['test_top5_accs'])
print(f'  最佳Top-1准确率: {best_top1:.2f}%')
print(f'  最佳Top-5准确率: {best_top5:.2f}%')
print('  ----------------------------------------')
"
    fi
done

echo ""
echo "实验完成时间: $(date)"
echo "==========================================" 