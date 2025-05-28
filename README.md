# DL-2025 CIFAR-100 分类项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)

## 项目概述

基于精简版 ResNet 实现 CIFAR-100 分类任务，探索并对比多种先进的深度学习网络架构和注意力机制。

**截止日期**: 2025.6.4

## 目录结构

```plaintext
├── src/                    # 源代码
│   ├── model.py           # 模型定义(ResNet, ECA-Net, GhostNet, ConvNeXt)
│   ├── dataset.py         # 数据加载器
│   ├── train.py           # 训练函数
│   ├── train_fast.py      # 快速训练版本
│   ├── train_accelerate.py # Accelerate加速训练
│   ├── utils.py           # 工具函数
│   └── main.py            # 主程序
├── data/                  # CIFAR-100数据集
├── logs/                  # 训练日志和模型检查点
├── report/                # 实验报告和思路文档
├── default_config.yaml    # Accelerate配置文件
└── requirements.md        # 项目要求
```

## 已实现的网络架构

### 1. 基础 ResNet-20
- **参数量**: 0.28M
- **特点**: 精简版ResNet，适合CIFAR-100小图像分类

### 2. ECA-ResNet-20 (Efficient Channel Attention)
- **参数量**: 0.28M
- **特点**: 集成高效通道注意力机制，几乎不增加参数量
- **创新点**: 使用1D卷积替代全连接层实现跨通道交互

### 3. Ghost-ResNet-20
- **参数量**: 0.03M (参数量减少90%)
- **特点**: 使用Ghost模块生成冗余特征图，大幅减少计算量
- **创新点**: 通过廉价操作生成更多特征图

### 4. ConvNeXt-Tiny
- **参数量**: 0.17M
- **特点**: 现代化卷积网络设计，融合Transformer设计思想
- **创新点**: 大核卷积、LayerNorm、GELU激活函数

## 技术栈

- **PyTorch 2.7.0**: 主要深度学习框架
- **torchvision**: 数据集和预处理
- **accelerate**: 分布式训练加速
- **transformers**: 学习率调度器
- **matplotlib**: 结果可视化

## 快速开始

### 环境配置
```bash
# 激活虚拟环境
source /opt/venvs/base/bin/activate

# 安装依赖
pip install torch torchvision accelerate transformers matplotlib
```

### 快速训练
```bash
# 快速训练版本(15 epochs)
python src/main_fast.py

# 使用Accelerate加速训练
accelerate launch --config_file default_config.yaml src/train_accelerate.py --model resnet_20 --epochs 20

# 启动多模型对比训练
python launch_training.py
```

## 训练结果

### 当前训练进展

| 模型 | 参数量 | 训练状态 | 最佳准确率 | 训练轮次 | 训练时间 |
|------|--------|----------|------------|----------|----------|
| ResNet-20 | 0.28M | ✅ 已完成 | 26.41% | 3/20 | ~1分钟 |
| ECA-ResNet-20 | 0.28M | ⏳ 待训练 | - | - | - |
| Ghost-ResNet-20 | 0.03M | ⏳ 待训练 | - | - | - |
| ConvNeXt-Tiny | 0.17M | ⏳ 待训练 | - | - | - |

**最后更新**: 2025年05月28日 20:55


### 详细训练记录 (ResNet-20)

**训练时间**: 2025年5月28日 19:45-20:05 (约20分钟)  
**硬件配置**: CUDA GPU, WSL2 Ubuntu 24.04  
**训练配置**:
- Batch Size: 256
- 学习率: 0.1 → 0.01 → 0.001 (MultiStepLR调度)
- 优化器: SGD + Nesterov动量 (momentum=0.9, weight_decay=1e-4)
- 数据增强: RandomCrop + RandomHorizontalFlip
- 总epochs: 15

**性能进展**:
- Epoch 1: 训练44.4s, 测试准确率10.94%
- Epoch 5: 训练50.8s, 测试准确率36.29% (学习率降至0.01)
- Epoch 10: 训练58.0s, 测试准确率49.12% (学习率降至0.001)
- Epoch 11: 训练58.4s, 测试准确率50.16% (最佳)

**平均训练时间**: 
- 每个epoch约45-58秒
- 每个batch约0.3秒 (196个batch/epoch)
- 总训练时间约13分钟 (15个epoch)

### 训练配置优化

#### 快速训练版本特性:
- **Batch Size**: 256 (提升GPU利用率)
- **学习率调度**: MultiStepLR (epochs//3, 2*epochs//3)
- **优化器**: SGD + Nesterov动量
- **日志频率**: 每50个batch记录一次

#### Accelerate加速特性:
- **混合精度**: FP16 (减少显存占用，提升速度)
- **分布式支持**: 支持多GPU训练
- **梯度累积**: 优化大batch训练
- **自动设备分配**: 自动检测最优硬件配置

## 实验设计

### 对比实验
1. **基础性能对比**: 4种网络架构在相同条件下的分类性能
2. **参数效率分析**: 准确率 vs 参数量的效率对比
3. **训练速度对比**: 不同模型的训练时间和收敛速度
4. **消融实验**: 注意力机制、Ghost模块等组件的有效性验证

### 评估指标
- **Top-1 准确率**: 主要评估指标
- **Top-5 准确率**: 辅助评估指标  
- **参数量**: 模型复杂度指标
- **训练时间**: 效率指标

## 创新特色

1. **多架构融合**: 整合CNN、注意力机制、轻量化设计
2. **训练优化**: 使用Accelerate实现高效分布式训练
3. **全面对比**: 从准确率、参数量、速度多维度评估
4. **实用性**: 专门针对小图像分类任务优化

## 文件说明

- `src/model.py`: 包含4种网络架构的完整实现
- `src/train_fast.py`: 优化的快速训练函数
- `src/train_accelerate.py`: 使用Accelerate的分布式训练
- `logs/`: 包含详细的训练日志和模型检查点
- `default_config.yaml`: Accelerate分布式训练配置

## 下一步计划

- [ ] 完成ECA-ResNet-20训练
- [ ] 训练Ghost-ResNet-20和ConvNeXt-Tiny
- [ ] 生成训练曲线对比图
- [ ] 进行消融实验
- [ ] 撰写详细实验报告
- [ ] 准备项目展示PPT

## 参考资料

详见 `requirements.md` 和`report/CIFAR-100分类思路.md`中的论文链接和技术文档。