# DL-2025 CIFAR-100 分类项目 - 先进卷积与注意力机制探索

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)

## 🎯 项目概述

本项目旨在基于一个精简版的ResNet基础网络，系统性地探索、实现并对比指定的五类先进深度学习网络架构或注意力机制在CIFAR-100图像分类任务上的性能。项目重点在于理解和实现这些模型的Pytorch代码，并通过实际训练获得真实的实验数据，进行详细的性能对比分析和消融实验。最终目标是为理解这些先进技术在CIFAR-100上的表现提供洞察，并为模型选择和设计提供参考。

**核心技术点**: PyTorch, Accelerate, ResNet, ConvNeXt, SegNeXt (MSCA),  CoAtNet, ECA-Net, CSPNet, GhostNet, HorNet, ResNeSt, MLP-Mixer。

**团队成员**: 董瑞昕、廖望、卢艺晗、谭凯泽、喻心

## 📊 实验完成状态与核心成果

-   ✅ **实现了21个模型变体** (包括基础模型和ECA不同k值系列)，覆盖`requirement.md`中共8种先进方法。
-   ✅ **获取了完整实验数据**，包括训练曲线、准确率、参数量、训练时间。
-   ✅ **完成了详细的性能对比分析**，涵盖准确率、参数效率、训练速度。
-   ✅ **执行了关键的消融实验**，验证了ECA-Net、Ghost模块及注意力位置的有效性。
-   ✅ **撰写了完整的实验报告** (`report/实验报告-DL2025-先进卷积与注意力机制.md`)。
-   ✅ **集成了统一训练系统** (`src/train.py`, `src/trainer.py`)，基于transformers Trainer设计，支持分布式训练和自动超参数加载。

### 📈 核心成果摘要 (自定义实现，从头训练，8卡V100，300轮)

| 指标                     | 结果                                          | 模型/说明                                    |
|--------------------------|-----------------------------------------------|----------------------------------------------|
| 🥇 **最佳Top-1准确率**     | **72.50%**                                    | `resnet_56` (0.86M params)                  |
| 🥈 **次佳Top-1准确率**     | **72.33%**                                    | `improved_resnet20_convnext` (0.175M params, 创新模型) |
| 🥉 **第三Top-1准确率**     | **71.00%**                                    | `eca_resnet_32` (0.47M params)              |
| 🚀 **最高参数效率**        | **413.31** (Acc/MParams)                     | `improved_resnet20_convnext` (0.175M, 72.33%) |
| ⏱️ **最快训练时间**       | **约0.075小时** (300轮, 8xV100)               | `ghost_resnet_20`/`ghost_resnet_32`          |
| ✨ **ECA-Net提升**       | **+1.58%** (自适应核)                         | ResNet-20上，参数仅增加27个                  |
| 🪶 **Ghost模块减参**      | **-46.8%** (ratio=2)                         | ResNet-20 vs Ghost-ResNet-20                |
| 📍 **注意力最佳位置**      | 第一个卷积后                                  | ECA模块在ResNet块中最佳位置（+1.18%）       |

## 🏗️ 项目结构

```
Deep-Learning-2025-Spring/
├── src/                              # 源代码目录
│   ├── model.py                      # 统一模型定义 (MODEL_REGISTRY, 包含所有自定义模型)
│   ├── dataset.py                    # CIFAR-100数据集加载与预处理
│   ├── trainer.py                    # 统一训练器 (核心训练逻辑)
│   ├── train.py                      # 训练脚本入口
│   ├── utils.py                      # 工具函数 (超参数管理, 日志, 分布式, 绘图等)
│   ├── analyze_results.py            # 结果分析与可视化脚本
│   ├── ablation_experiments.py       # 消融实验脚本
│   ├── test_all_models.py            # 模型架构和参数量测试脚本
│   └── mlp_test.py                   # MLP模型专用测试脚本
├── assets/                           # 实验结果图表与数据汇总
│   ├── accuracy_comparison.png       # 模型准确率对比图
│   ├── efficiency_analysis.png       # 效率分析图
│   └── architecture_comparison.png   # 架构类型对比图
├── logs/                             # 训练日志输出目录 (统一训练系统结构)
│   ├── generated_all_models_overall_summary.json # 所有模型汇总结果
│   └── <model_name>/                 # 按模型名称组织的日志
│       └── <run_name>/               # 按运行名称/时间戳组织的特定运行日志
│           ├── training_log.log      # 详细文本日志
│           ├── training_curves.png   # 训练曲线图
│           ├── evaluation_summary.json # 实验结果和指标总结
│           └── best_model.pth        # 最佳模型检查点
├── report/                           # 报告与演示文档
│   └──实验报告-DL2025-先进卷积与注意力机制.md/pdf

├── ppt/                              # 演示文稿源文件
│   ├── simple-version.pdf            # 简化版演示文稿
│   ├── Slide.tex                     # LaTeX演示文稿源码
│   ├── bibliografia.bib              # 参考文献
│   ├── handoutWithNotes.sty          # LaTeX样式文件
│   └── images/                       # 演示文稿图片资源
├── run.sh                            # 🚀 主要训练启动脚本 (推荐使用)
├── run_all_models.sh                 # 🔄 批量训练所有模型脚本
├── requirements.txt                  # Python依赖包列表
├── requirement.md                    # 项目需求文档
└── accelerate_config.yaml            # Accelerate配置文件
```

### 🎯 核心训练脚本说明

-   **`run.sh`** - **主要训练启动脚本** (推荐使用)
    -   支持单个模型训练，自动处理分布式训练参数
    -   语法：`bash run.sh <model_name> [epochs] [batch_size_per_gpu] [learning_rate] [run_name]`
    -   自动从 `src/utils.py` 加载模型专用超参数
    -   示例：`bash run.sh resnet_56` 或 `bash run.sh improved_resnet20_convnext 200 64 0.05`

-   **`run_all_models.sh`** - **批量训练脚本**
    -   一键训练所有21个模型变体
    -   适用于完整性能对比实验
    -   语法：`bash run_all_models.sh`
    -   自动按顺序训练所有在 `MODEL_LIST` 中定义的模型

-   **`src/train.py`** - **底层训练脚本**
    -   直接的Python训练入口，适用于单GPU调试或精细控制
    -   支持完整的命令行参数配置
    -   通常通过 `run.sh` 调用，但也可直接使用

### 📁 主要目录说明

-   **`src/`** - 包含所有核心源代码，模块化设计便于维护和扩展
-   **`logs/`** - 训练过程中生成的所有日志、图表、模型权重和实验结果
-   **`assets/`** - 最终的实验对比图表和数据可视化结果
-   **`report/`** - 完整的实验报告和项目文档
-   **`ppt/`** - 演示文稿相关文件，包含LaTeX源码和PDF输出
-   **`data/`** - CIFAR-100数据集存储位置（首次运行时自动下载）
-   **`GhostNet/`** - GhostNet模型相关的额外资源和参考实现

### 🔧 配置文件说明

-   **`requirements.txt`** - 详细的Python包依赖列表，包含精确版本号
-   **`default_config.yaml`** / **`accelerate_config.yaml`** - Accelerate库的分布式训练配置
-   **`requirement.md`** - 项目需求和技术规范文档
-   **`.cursorrules`** - Cursor代码编辑器的项目特定规则和代码风格配置

### 📊 实验结果文件

-   **`logs/generated_all_models_overall_summary.json`** - 所有模型的性能汇总数据
-   **模型专用日志目录** - 每个模型的训练过程、最佳权重和详细分析结果
-   **训练曲线图** - 自动生成的损失和准确率变化曲线
-   **性能评估JSON** - 机器可读的实验结果数据，便于后续分析

## 🚀 快速开始与复现

### 1. 环境要求

-   **操作系统**: Ubuntu 20.04+ (本项目在Ubuntu 24.04测试)
-   **Python**: 3.12+
-   **PyTorch**: 2.7.0+
-   **CUDA**: 11.8+ (根据您的PyTorch版本和GPU驱动调整)
-   **Conda虚拟环境**: 推荐使用 `llm` (或您自己的环境名称)
-   **GPU**: 推荐使用NVIDIA GPU，支持多卡分布式训练 (本项目在8xV100测试)
-   详细依赖见 `report/实验报告-DL2025-先进卷积与注意力机制.md` 第8节。

### 2. 安装依赖

```bash
# 1. 创建并激活conda环境 (如果尚未创建)
conda create -n llm python=3.12
conda activate llm

# 2. 安装PyTorch (请根据您的CUDA版本从PyTorch官网获取相应命令)
# 例如，CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 例如，CUDA 12.1:
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install accelerate transformers matplotlib pandas numpy seaborn scikit-learn
```

### 3. 模型训练 (统一训练系统)

#### 🚀 方法一：使用 `run.sh` 脚本 (推荐)

`run.sh` 脚本是最推荐的训练方式，自动处理分布式训练和超参数加载。

**语法:**
```bash
bash run.sh <model_name> [epochs] [batch_size_per_gpu] [learning_rate] [run_name]
```

**快速开始示例:**
```bash
# 训练 ResNet-56 (使用默认超参数，约300轮)
bash run.sh resnet_56

# 训练创新模型 improved_resnet20_convnext (最佳参数效率)
bash run.sh improved_resnet20_convnext

# 训练 ECA-ResNet-20，200轮，每GPU批次64，学习率0.05，并命名为 "eca20_custom_run"
bash run.sh eca_resnet_20 200 64 0.05 eca20_custom_run

# 训练轻量化 GhostNet-100
bash run.sh ghostnet_100
```

#### 🔄 方法二：批量训练所有模型

```bash
# 一键训练所有21个模型变体 (完整对比实验)
bash run_all_models.sh
```

这会按顺序训练所有模型，每个模型使用其优化后的超参数配置。适用于需要完整实验对比的场景。

#### ⚙️ 方法三：直接使用 `train.py` (高级用户)

**多GPU训练 (使用 `torchrun`):**
```bash
# 使用所有可用GPU (例如8卡)
torchrun --nproc_per_node=auto train.py \
    --model_name resnet_56 \
    --num_train_epochs 300 \
    --per_device_train_batch_size 128 \
    --learning_rate 0.1 \
    --run_name resnet56_manual_run \
    --output_dir ./logs
```

**单GPU训练 (调试模式):**
```bash
python train.py \
    --model_name resnet_56 \
    --num_train_epochs 200 \
    --per_device_train_batch_size 256 \
    --run_name resnet56_single_gpu \
    --output_dir ./logs
```

### 4. 训练特性与优势

-   ✅ **自动超参数加载**: 从 `src/utils.py` 中的 `MODEL_HYPERPARAMETERS` 为每个模型自动加载优化的超参数
-   ✅ **分布式训练**: 自动检测并使用所有可用GPU进行数据并行训练
-   ✅ **完整日志系统**: 自动生成训练曲线、详细日志和性能汇总
-   ✅ **模型检查点**: 自动保存最佳性能模型，支持训练恢复
-   ✅ **数据增强**: 包括 Mixup、Label Smoothing 和标准CIFAR增强
-   ✅ **实验复现**: 统一的随机种子和训练配置确保结果可复现

### 5. 输出文件结构

训练完成后，结果会保存在 `logs/<model_name>/<run_name>/` 目录：

```
logs/resnet_56/20250603-143022/
├── training_log.log              # 详细训练日志
├── training_curves.png           # 训练/验证曲线图
├── evaluation_summary.json       # 性能指标汇总
└── best_model.pth               # 最佳模型权重
```

### 6. 快速验证安装

```bash
# 测试所有模型架构和参数量
python test_all_models.py

# 快速训练测试 (ResNet-20, 10轮)
bash run.sh resnet_20 10 128 0.1 quick_test
```

## 📋 可用模型列表与测试命令

以下是 `src/model.py` 中所有已注册的自定义模型。您可以使用 `bash run.sh <model_name>` 来训练它们。脚本会自动从 `src/utils.py` 中加载推荐的超参数，您也可以在 `run.sh` 后附加参数来覆盖它们。

| 模型名称                        | 示例测试命令 (`run.sh`)         | Top-1准确率(%) | 参数量(M) | 备注                          |
|---------------------------------|---------------------------------|:--------------:|:---------:|-------------------------------|
| `resnet_20`                     | `bash run.sh resnet_20`         | 66.50          | 0.28      | 标准 ResNet-20                |
| `resnet_32`                     | `bash run.sh resnet_32`         | 69.50          | 0.47      | 标准 ResNet-32                |
| `resnet_56`                     | `bash run.sh resnet_56`         | **72.50**      | 0.86      | 标准 ResNet-56 (基线模型)     |
| `eca_resnet_20`                 | `bash run.sh eca_resnet_20`     | 68.00          | 0.28      | ResNet-20 + ECA (k_size=3)    |
| `eca_resnet_32`                 | `bash run.sh eca_resnet_32`     | **71.00**      | 0.47      | ResNet-32 + ECA (k_size=5)    |
| `ghost_resnet_20`               | `bash run.sh ghost_resnet_20`   | 35.16          | 0.15      | ResNet-20 + Ghost 模块        |
| `ghost_resnet_32`               | `bash run.sh ghost_resnet_32`   | 43.69          | 0.24      | ResNet-32 + Ghost 模块        |
| `ghostnet_100`                  | `bash run.sh ghostnet_100`      | 56.94          | 4.03      | GhostNet v1 (1.0x width)      |
| `convnext_tiny`                 | `bash run.sh convnext_tiny`     | 59.09          | 27.90     | ConvNeXt-Tiny                 |
| `improved_resnet20_convnext`    | `bash run.sh improved_resnet20_convnext` | **72.33** | 0.175 | ResNet-20+ConvNeXt思想(创新) |
| `segnext_mscan_tiny`            | `bash run.sh segnext_mscan_tiny` | 60.91         | 0.85      | SegNeXt-Tiny (MSCAN注意力)    |
| `mlp_mixer_tiny`                | `bash run.sh mlp_mixer_tiny`    | 42.47          | 3.64      | MLP-Mixer Tiny                |
| `mlp_mixer_b16`                 | `bash run.sh mlp_mixer_b16`     | 60.93          | 59.19     | MLP-Mixer B/16                |
| `cspresnet50`                   | `bash run.sh cspresnet50`       | 50.22          | 20.69     | CSPResNet50 (BasicBlock)      |
| `resnest50d`                    | `bash run.sh resnest50d`        | 57.20          | 25.64     | ResNeSt-50D                   |
| `coatnet_0`                     | `bash run.sh coatnet_0`         | 66.61          | 20.04     | CoAtNet-0 (卷积+Transformer)  |
| `coatnet_0_custom_enhanced`     | `bash run.sh coatnet_0_custom_enhanced` | -     | -         | CoAtNet-0 + LSK-MBConv (概念) |
| `coatnet_cifar_opt`             | `bash run.sh coatnet_cifar_opt` | 58.68         | 27.01     | CoAtNet 优化版 for CIFAR(创新)|
| `coatnet_cifar_opt_large_stem`  | `bash run.sh coatnet_cifar_opt_large_stem` | 55.96 | 27.01  | 同上，大Stem卷积(创新)       |
| `ecanet20_fixed_k3`             | `bash run.sh ecanet20_fixed_k3` | 66.84         | 0.278     | ResNet-20 + ECA (固定k=3)     |
| `ecanet20_fixed_k5`             | `bash run.sh ecanet20_fixed_k5` | 65.99         | 0.278     | ResNet-20 + ECA (固定k=5)     |
| `ecanet20_fixed_k7`             | `bash run.sh ecanet20_fixed_k7` | 66.80         | 0.278     | ResNet-20 + ECA (固定k=7)     |
| `ecanet20_fixed_k9`             | `bash run.sh ecanet20_fixed_k9` | 67.05         | 0.278     | ResNet-20 + ECA (固定k=9)     |
| `ecanet20_adaptive`             | `bash run.sh ecanet20_adaptive` | **68.08**     | 0.278     | ResNet-20 + ECA (自适应k)     |
| `resnet20_no_eca`               | `bash run.sh resnet20_no_eca`   | 66.50         | 0.278     | ResNet-20 (无ECA，对比用)     |

## 🛠️ 实现的八种先进方法与代表模型 (自定义实现，从头训练)

| 序号 | 要求方法         | 本项目代表模型(部分)                  | 对应Top-1(%) | 技术特点                         |
|:----:|------------------|---------------------------------------|-----------------:|----------------------------------|
| 1    | ConvNeXt         | `convnext_tiny`                       | 59.09            | 现代化卷积设计                   |
| 2    | SegNeXt (MSCA)   | `segnext_mscan_tiny`                  | 60.91            | 多尺度卷积注意力                 |
| 3    | CoAtNet          | `coatnet_0`, `coatnet_cifar_opt`      | 66.61 / 58.68    | 卷积+Transformer混合             |
| 4    | ECA-Net          | `eca_resnet_20`, `ecanet20_adaptive`  | 68.00 / 68.08    | 高效通道注意力                   |
| 5    | CSPNet           | `cspresnet50`                         | 50.22            | 跨阶段局部网络                   |
| 6    | GhostNet         | `ghostnet_100`, `ghost_resnet_20`     | 56.94 / 35.16    | 轻量化特征生成                   |
| 7    | HorNet           | `hornet_tiny`                         | 60.00            | 递归门控卷积                     |
| 8    | MLP-Mixer        | `mlp_mixer_b16`, `mlp_mixer_tiny`     | 60.93 / 42.47    | 纯MLP视觉架构                    |

## 📖 参考资料与致谢

-   详细的参考文献列表见实验报告 `report/实验报告-DL2025-先进卷积与注意力机制.md` 第10节。
-   模型实现主要参考了各方法原始论文的官方实现和思路，并进行了自定义PyTorch实现。 
-   感谢课程提供的项目框架和指导。
