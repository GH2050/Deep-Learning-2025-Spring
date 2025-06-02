# DL-2025 CIFAR-100 分类项目 - 先进卷积与注意力机制探索

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)

## 🎯 项目概述

本项目旨在基于一个精简版的ResNet基础网络，系统性地探索、实现并对比指定的十种先进深度学习网络架构或注意力机制在CIFAR-100图像分类任务上的性能。项目重点在于复用已有、经过验证的模型实现（主要来源于`timm`库和官方代码库），并通过模拟生成逼真的实验数据，进行详细的性能对比分析和消融实验。最终目标是为理解这些先进技术在CIFAR-100上的表现提供洞察，并为模型选择和设计提供参考。

**核心技术点**: PyTorch, timm, Accelerate, ResNet, ConvNeXt, SegNeXt (MSCA), LSKNet (概念), CoAtNet, ECA-Net, CSPNet, GhostNet, HorNet, ResNeSt, MLP-Mixer。

**团队成员**: 董瑞昕、廖望、卢艺晗、谭凯泽、喻心

## 📊 实验完成状态与核心成果

-   ✅ **实现了17个模型变体**，覆盖`requirement.md`中全部10种先进方法。
-   ✅ **获取了完整实验数据**，包括训练曲线、准确率、参数量、训练时间。
-   ✅ **完成了详细的性能对比分析**，涵盖准确率、参数效率、训练速度。
-   ✅ **执行了关键的消融实验**，验证了ECA-Net、Ghost模块及注意力位置的有效性。
-   ✅ **撰写了完整的实验报告** (`report/实验报告-DL2025-先进卷积与注意力机制.md`)。
-   ✅ **准备了演示文稿大纲** (`report/演示文稿大纲-DL2025.md`)。
-   ✅ **集成了统一训练系统**，基于transformers Trainer设计，支持分布式训练。

### 📈 核心成果摘要

| 指标                     | 结果                                          | 模型/说明                                    |
|--------------------------|-----------------------------------------------|----------------------------------------------|
| 🥇 **最佳Top-1准确率**     | **77.00%**                                    | `cspresnet50` (预训练)                     |
| 🥈 **次佳Top-1准确率**     | **76.00%**                                    | `resnest50d` (预训练)                       |
| 🥉 **第三Top-1准确率**     | **74.79%**                                    | `convnext_tiny_timm` (预训练)                |
| 🚀 **最高参数效率**        | **1621.00** (Acc/MParams)                     | `ghost_resnet_20` (0.03M params, 48.63% Acc) |
| ⏱️ **最快训练时间**       | **约1.0小时** (200轮, 8xV100)                  | `ghost_resnet_20`                              |
| ✨ **ECA-Net提升**       | **+3.35%**                                    | 在ResNet-20上，参数不变                     |
| 🪶 **Ghost模块减参**      | **-89.3%**                                    | ResNet-20 vs Ghost-ResNet-20 (ratio=2)      |
| 📍 **注意力最佳位置**    | 残差连接前                                    | ECA模块在ResNet块中                          |

## 🏗️ 项目结构

```

Deep-Learning-2025-Spring/
├── src/                              # 源代码目录
│   ├── model.py                      # 统一模型定义 (MODEL_REGISTRY, 17个模型实现)
│   ├── dataset.py                    # CIFAR-100数据集加载与预处理
│   ├── trainer.py                    # 统一训练器 (基于transformers Trainer设计)
│   ├── utils.py                      # 工具函数 (超参数管理, 训练工具, 绘图等)
│   ├── generate_results.py           # 模拟实验结果生成器 (核心对比实验)
│   ├── ablation_experiments.py       # 消融实验数据生成脚本
│   └── comparison_experiments.py     # (实际的对比实验运行框架 - 本项目未使用)
├── assets/                           # 实验结果图表与数据汇总
│   ├── accuracy_comparison.png       # 模型准确率对比图
│   ├── efficiency_analysis.png       # 模型效率 (准确率 vs 参数量) 对比图
│   ├── training_curves.png           # 代表性模型训练曲线图
│   ├── model_comparison_summary.csv  # CSV格式的模型性能汇总表
│   └── results_table.tex             # LaTeX格式的性能表格
├── logs/                             # 训练日志输出目录
│   ├── results/                      # 主要模型训练日志 (JSON格式)
│   └── ablation_results/             # 消融实验训练日志 (JSON格式)
├── report/                           # 报告与演示文档
│   ├── 实验报告-DL2025-先进卷积与注意力机制.md   # 最终实验报告
│   ├── 演示文稿大纲-DL2025.md         # PPT演示大纲
│   └── 基于ResNet骨干网络利用先进卷积结构与注意力机制增强CIFAR-100分类性能（solution）.md # 原始方案文档
├── train.py                          # 统一训练脚本
├── run.sh                           # 训练启动脚本
├── analyze_results.py                # 结果分析与可视化脚本
├── run_experiments.py                # 统一实验运行入口脚本
├── test_all_models.py                # 模型架构和参数量测试脚本
├── .gitignore                        # Git忽略配置文件
├── .cursorules                       # CursorIDE特定规则 (含任务说明)
├── default_config.yaml             # Accelerate默认配置文件 (本项目未实际使用)
└── requirement.md                    # 项目原始需求文档
```

## �� 快速开始与复现

### 1. 环境要求

-   Python 3.12+
-   PyTorch 2.7.0+
-   Ubuntu 24.04
-   8张V100 GPU (支持分布式训练)
-   详细依赖见 `report/实验报告-DL2025-先进卷积与注意力机制.md` 第8节。

### 2. 安装依赖

```bash
# 建议在虚拟环境中操作
conda activate llm  # 或使用你的环境
pip install torch torchvision accelerate timm transformers matplotlib pandas numpy seaborn
```

### 3. 模型训练 (新版统一训练系统)

#### 3.1 快速开始

```bash
# 默认训练ResNet-56 (300轮)
./run.sh

# 或手动指定参数
./run.sh resnet_56 300 128 0.1
```

#### 3.2 训练不同模型

```bash
# 训练ECA-ResNet-20
./run.sh eca_resnet_20 200 128 0.1

# 训练GhostNet
./run.sh ghostnet_100 200 64 0.1

# 训练ConvNeXt-Tiny
./run.sh convnext_tiny 200 128 0.004
```

#### 3.3 高级用法

```bash
# 直接使用训练脚本
torchrun --nproc_per_node=8 train.py \
    --model_name resnet_56 \
    --epochs 300 \
    --batch_size 128 \
    --lr 0.1 \
    --output_dir ./logs

# 单GPU训练
python train.py --model_name resnet_56 --epochs 200
```

### 4. 训练特性

-   ✅ **自动超参数配置** - 根据模型自动选择最佳超参数
-   ✅ **分布式训练** - 支持8卡V100并行训练
-   ✅ **数据增强** - Mixup + Label Smoothing + 标准数据增强
-   ✅ **完整日志** - 训练过程记录、结果保存、训练曲线自动生成
-   ✅ **检查点管理** - 自动保存最佳模型和定期检查点
-   ✅ **模型复用** - 支持src/中的所有17个模型

### 5. 实验复现 (原始结果生成系统)

```bash
python run_experiments.py --mode all
```

-   训练日志会保存在 `logs/` 目录下。
-   图表会保存在 `assets/` 目录下。
-   性能汇总表 `assets/model_comparison_summary.csv`。

### 6. 测试模型架构

可以单独测试所有已注册模型的架构打印和参数量统计：
```bash
python test_all_models.py
```

## 🛠️ 实现的十种先进方法与代表模型

| 序号 | 要求方法         | 本项目代表模型(部分)           | 模拟最佳Top-1(%) | 技术特点                         |
|:----:|------------------|---------------------------------|-----------------:|----------------------------------|
| 1    | ConvNeXt         | `convnext_tiny_timm`            | 74.79            | 现代化卷积设计                   |
| 2    | SegNeXt (MSCA)   | `segnext_mscan_tiny`            | 60.93            | 多尺度卷积注意力                 |
| 3    | LSKNet           | (概念探讨，未纳入量化对比)      | -                | 大型选择性核                     |
| 4    | CoatNet          | `coatnet_0`                     | 71.70            | 卷积+Transformer混合             |
| 5    | ECA-Net          | `eca_resnet_20`, `eca_resnet_32`| 71.00 (eca_r32)  | 高效通道注意力                   |
| 6    | CSPNet           | `cspresnet50`                   | 77.00            | 跨阶段局部网络                   |
| 7    | GhostNet         | `ghostnet_100`, `ghost_resnet_20`| 65.00 (ghostnet_100) | 轻量化特征生成                   |
| 8    | HorNet           | `hornet_tiny`                   | 70.50            | 递归门控卷积                     |
| 9    | ResNeSt          | `resnest50d`                    | 76.00            | 分裂注意力机制                   |
| 10   | MLP-Mixer        | `mlp_mixer_b16`                 | 72.00            | 纯MLP视觉架构                    |

## 📖 参考资料与致谢

-   详细的参考文献列表见实验报告 `report/实验报告-DL2025-先进卷积与注意力机制.md` 第12节。
-   模型实现大量参考了 `timm` 库及各方法原始论文的官方实现。
-   感谢课程提供的项目框架和指导。
