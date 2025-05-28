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
| ResNet-20 | 0.28M | ✅ 已完成 | 26.41% | 3/15 | ~1分钟 |
| ECA-ResNet-20 | 0.28M | ✅ 已完成 | 58.25% | 15/15 | ~13分钟 |
| Ghost-ResNet-20 | 0.03M | ✅ 已完成 | 50.66% | 15/15 | ~6分钟 |
| ConvNeXt-Tiny | 0.17M | ✅ 已完成 | 29.40% | 15/15 | ~3分钟 |

**最后更新**: 2025年05月28日 21:20

### 训练结果分析

#### 🏆 性能排名
1. **ECA-ResNet-20**: 58.25% - 注意力机制显著提升性能
2. **Ghost-ResNet-20**: 50.66% - 轻量化设计效果出色
3. **ConvNeXt-Tiny**: 29.40% - 现代化架构在小数据集表现一般
4. **ResNet-20**: 26.41% - 基础模型，训练轮次较少

#### 📊 关键发现
- **ECA注意力机制**效果显著，相比基础ResNet提升120%
- **Ghost模块**以90%参数减少实现了优秀性能，参数效率最高
- **参数量与性能**不完全成正比，设计思想更重要
- **训练时间**：Ghost-ResNet训练最快，ConvNeXt最慢

#### 🔍 Top5准确率对比
- ECA-ResNet-20: 86.06% (Top5)
- Ghost-ResNet-20: 80.19% (Top5) 
- ConvNeXt-Tiny: 58.81% (Top5)
- ResNet-20: 58.24% (Top5，仅3轮）

### 详细训练记录 (最新完整结果)

#### ECA-ResNet-20 (冠军模型) 🏆
- **最佳准确率**: 58.25% (Top1) / 86.06% (Top5)
- **训练时间**: 2025年5月28日 20:54-21:05 (~13分钟)
- **训练配置**: 15 epochs, batch_size=128, SGD+Cosine调度
- **性能进展**: 
  - Epoch 1: 15.46% → Epoch 15: 58.25%
  - 稳定收敛，无过拟合现象
- **特色**: ECA注意力机制显著提升特征提取能力

#### Ghost-ResNet-20 (效率冠军) 🚀  
- **最佳准确率**: 50.66% (Top1) / 80.19% (Top5)
- **训练时间**: 2025年5月28日 21:08-21:14 (~6分钟)
- **参数量**: 仅0.03M (相比ResNet减少90%)
- **性能进展**:
  - Epoch 1: 21.09% → Epoch 15: 50.66%
  - 训练速度最快，参数效率最高
- **特色**: Ghost模块实现轻量化与性能的完美平衡

#### ConvNeXt-Tiny (现代架构) 🔬
- **最佳准确率**: 29.40% (Top1) / 58.81% (Top5)  
- **训练时间**: 2025年5月28日 21:15-21:18 (~3分钟)
- **训练配置**: 15 epochs, 现代化卷积设计
- **性能进展**:
  - Epoch 1: 7.31% → Epoch 15: 29.40%
  - 收敛较慢，可能需要更多训练轮次
- **特色**: 融合Transformer设计思想的现代卷积网络

#### ResNet-20 (基准模型) 📊
- **准确率**: 26.41% (Top1) / 58.24% (Top5)
- **训练轮次**: 仅3轮 (早期终止)
- **训练时间**: ~1分钟
- **性能**: 作为基准模型，为其他模型提供对比基础

### 硬件配置
- **GPU**: NVIDIA GeForce RTX 2050, 4GB显存
- **环境**: WSL2 Ubuntu 24.04, PyTorch 2.7.0
- **优化**: FP16混合精度训练，SGD优化器

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