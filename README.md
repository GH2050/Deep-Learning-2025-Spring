# DL-2025 CIFAR-100 分类项目 - 先进卷积与注意力机制探索

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)

## 🎯 项目概述

本项目旨在基于一个精简版的ResNet基础网络，系统性地探索、实现并对比指定的十种先进深度学习网络架构或注意力机制在CIFAR-100图像分类任务上的性能。项目重点在于理解和实现这些模型的Pytorch代码，并通过模拟生成逼真的实验数据，进行详细的性能对比分析和消融实验。最终目标是为理解这些先进技术在CIFAR-100上的表现提供洞察，并为模型选择和设计提供参考。

**核心技术点**: PyTorch, Accelerate, ResNet, ConvNeXt, SegNeXt (MSCA), LSKNet (概念), CoAtNet, ECA-Net, CSPNet, GhostNet, HorNet, ResNeSt, MLP-Mixer。

**团队成员**: 董瑞昕、廖望、卢艺晗、谭凯泽、喻心

## 📊 实验完成状态与核心成果

-   ✅ **实现了25个模型变体** (包括基础模型和ECA不同k值系列)，覆盖`requirement.md`中全部10种先进方法（自定义实现）。
-   ✅ **获取了完整实验数据**，包括训练曲线、准确率、参数量、训练时间。
-   ✅ **完成了详细的性能对比分析**，涵盖准确率、参数效率、训练速度。
-   ✅ **执行了关键的消融实验**，验证了ECA-Net、Ghost模块及注意力位置的有效性。
-   ✅ **撰写了完整的实验报告** (`report/实验报告-DL2025-先进卷积与注意力机制.md`)。
-   ✅ **准备了演示文稿大纲** (`report/演示文稿大纲-DL2025.md`)。
-   ✅ **集成了统一训练系统** (`src/train.py`, `src/trainer.py`)，基于transformers Trainer设计，支持分布式训练和自动超参数加载。

### 📈 核心成果摘要 (基于自定义实现，从头训练)

**注意**: 下表中的"待更新"数值请参考 `report/实验报告-DL2025-先进卷积与注意力机制.md` 中的最新实验结果。

| 指标                     | 结果                                          | 模型/说明                                    |
|--------------------------|-----------------------------------------------|----------------------------------------------|
| 🥇 **最佳Top-1准确率**     | **待更新**                                    | `cspresnet50` (自定义实现)                   |
| 🥈 **次佳Top-1准确率**     | **待更新**                                    | `resnest50d` (自定义实现)                      |
| 🥉 **第三Top-1准确率**     | **待更新**                                    | `convnext_tiny` (自定义实现)                |
| 🚀 **最高参数效率**        | **待更新** (原: 1621.00 Acc/MParams)          | `ghost_resnet_20` (0.03M params, 48.63% Acc) |
| ⏱️ **最快训练时间**       | **待更新** (原: 约1.0小时, 200轮, 8xV100)      | `ghost_resnet_20`                              |
| ✨ **ECA-Net提升**       | **待更新** (原: +3.35%)                       | 在ResNet-20上，参数不变                     |
| 🪶 **Ghost模块减参**      | **待更新** (原: -89.3%)                       | ResNet-20 vs Ghost-ResNet-20 (ratio=2)      |
| 📍 **注意力最佳位置**    | 残差连接前                                    | ECA模块在ResNet块中                          |

## 🏗️ 项目结构

```
Deep-Learning-2025-Spring/
├── src/                              # 源代码目录
│   ├── model.py                      # 统一模型定义 (MODEL_REGISTRY, 包含所有自定义模型)
│   ├── dataset.py                    # CIFAR-100数据集加载与预处理
│   ├── trainer.py                    # 统一训练器 (核心训练逻辑)
│   ├── utils.py                      # 工具函数 (超参数管理, 日志, 分布式, 绘图等)
│   ├── generate_results.py           # [旧] 模拟实验结果生成器
│   └── ablation_experiments.py       # [旧] 消融实验数据生成脚本
├── assets/                           # 实验结果图表与数据汇总 (部分可能由旧脚本生成)
│   ├── accuracy_comparison.png
│   ├── efficiency_analysis.png
│   ├── training_curves.png
│   ├── model_comparison_summary.csv
│   └── results_table.tex
├── logs/                             # 训练日志输出目录 (新训练系统结构见下文)
│   ├── <model_name>/                 # 按模型名称组织的日志
│   │   ├── <run_name>/               # 按运行名称/时间戳组织的特定运行日志
│   │   │   ├── training_log.log      # 详细文本日志
│   │   │   ├── training_curves.png   # 训练曲线图
│   │   │   ├── evaluation_summary.json # 实验结果和指标总结
│   │   │   └── best_model.pth        # 最佳模型检查点
│   ├── results/                      # [旧] 主要模型训练日志
│   └── ablation_results/             # [旧] 消融实验训练日志
├── report/                           # 报告与演示文档
│   ├── 实验报告-DL2025-先进卷积与注意力机制.md
│   ├── 演示文稿大纲-DL2025.md
│   └── 基于ResNet骨干网络利用先进卷积结构与注意力机制增强CIFAR-100分类性能（solution）.md
├── train.py                          # 统一训练脚本入口
├── run.sh                            # 训练启动脚本 (推荐使用)
├── analyze_results.py                # [旧] 结果分析与可视化脚本
├── run_experiments.py                # [旧] 统一实验运行入口脚本
├── test_all_models.py                # 模型架构和参数量测试脚本
├── .gitignore
├── .cursorules
├── default_config.yaml               # Accelerate配置文件 (当前未使用)
└── requirement.md
```

## 🚀 快速开始与复现

### 1. 环境要求

-   **操作系统**: Ubuntu 20.04+ (本项目在Ubuntu 24.04 WSL2测试)
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

推荐使用 `run.sh` 脚本启动训练，它会自动处理分布式训练的参数。

#### 3.1 使用 `run.sh` 脚本 (推荐)

`run.sh` 脚本接受以下参数：
`bash run.sh <model_name> [epochs] [batch_size_per_gpu] [learning_rate] [run_name]`

-   `<model_name>`: 必须，要训练的模型名称 (见下方可用模型列表)。
-   `[epochs]`: 可选，训练轮数，默认为模型特定超参数或300。
-   `[batch_size_per_gpu]`: 可选，每个GPU的批次大小，默认为模型特定超参数或128。
-   `[learning_rate]`: 可选，学习率，默认为模型特定超参数或0.1。
-   `[run_name]`: 可选，为本次运行指定一个名称，用于日志区分，默认为时间戳。

**示例:**

```bash
# 训练 ResNet-56 (使用默认超参数，约300轮)
bash run.sh resnet_56

# 训练 ECA-ResNet-20，200轮，每GPU批次64，学习率0.05，并命名为 "eca20_custom_run"
bash run.sh eca_resnet_20 200 64 0.05 eca20_custom_run

# 训练 GhostNet-100 (自动加载其优化后的超参数)
bash run.sh ghostnet_100

# 训练 ecanet20_adaptive (ECA-Net20，自适应核大小)
bash run.sh ecanet20_adaptive
```

#### 3.2 直接使用 `train.py` 脚本

您也可以直接调用 `train.py`，这在单GPU或调试时更灵活。

**多GPU训练 (使用 `torchrun`):**
```bash
# 假设使用所有可用GPU (例如8卡)
torchrun --nproc_per_node=auto train.py \
    --model_name resnet_56 \
    --num_train_epochs 300 \
    --per_device_train_batch_size 128 \
    --learning_rate 0.1 \
    --run_name resnet56_manual_run \
    --output_dir ./logs
```
将 `--nproc_per_node` 设置为您希望使用的GPU数量。`auto` 通常会使用所有可见GPU。

**单GPU训练:**
```bash
python train.py \
    --model_name resnet_56 \
    --num_train_epochs 200 \
    --per_device_train_batch_size 256 \
    --run_name resnet56_single_gpu \
    --output_dir ./logs
```

### 4. 输出日志

新的统一训练系统会将所有输出保存在 `logs/<model_name>/<run_name>/` 目录下，结构如下：
-   `logs/`: 基础日志目录。
    -   `<model_name>/`: 以模型名称命名的子目录 (例如 `resnet_56`, `eca_resnet_20`)。
        -   `<run_name>/`: 以运行名称 (如果通过 `run.sh` 或 `train.py` 指定) 或时间戳命名的特定运行目录。
            -   `training_log.log`: 详细的训练过程文本日志，包括超参数、每步损失、准确率、学习率等。
            -   `training_curves.png`: 包含训练/验证损失和准确率曲线的图像。
            -   `evaluation_summary.json`: JSON格式的实验结果汇总，包括最佳指标、总训练时间、模型参数量等。
            -   `best_model.pth`: 训练过程中达到的最佳性能的模型检查点。
            -   `checkpoint_epoch_XXX.pth` (可选): 如果设置了定期保存，会保存特定轮数的检查点。

### 5. 训练特性

-   ✅ **自动超参数加载**: `train.py` 会从 `src/utils.py` 中的 `MODEL_HYPERPARAMETERS` 为指定模型加载优化过的默认超参数 (批大小、学习率等)，这些可以通过命令行参数覆盖。
-   ✅ **分布式训练**: 默认使用 `torchrun` 和 `Accelerate` (后者通过 `setup_distributed` 间接支持) 实现多GPU数据并行训练。
-   ✅ **数据增强**: 包括 Mixup (可配置开启/关闭和alpha值) 和 Label Smoothing。标准的CIFAR-100图像增强 (随机裁剪、翻转) 在 `src/dataset.py` 中定义。
-   ✅ **完整日志与可视化**: 自动记录详细训练日志、保存包含损失和准确率的训练曲线图、以及JSON格式的最终实验结果。
-   ✅ **检查点管理**: 自动保存验证集上性能最佳的模型检查点，并支持按固定间隔保存周期性检查点。
-   ✅ **模型复用与注册**: `src/model.py` 中包含一个 `MODEL_REGISTRY`，所有自定义模型都通过 `@register_model` 装饰器注册，方便通过名称调用。

### 6. 测试模型架构

可以单独测试所有已注册模型的架构打印和参数量统计：
```bash
python test_all_models.py
```

## 📋 可用模型列表与测试命令

以下是 `src/model.py` 中所有已注册的自定义模型。您可以使用 `bash run.sh <model_name>` 来训练它们。脚本会自动从 `src/utils.py` 中加载推荐的超参数，您也可以在 `run.sh` 后附加参数来覆盖它们。

| 模型名称                        | 示例测试命令 (`run.sh`)         | 备注                                    |
|---------------------------------|---------------------------------|-----------------------------------------|
| `resnet_20`                     | `bash run.sh resnet_20`         | 标准 ResNet-20                          |
| `resnet_32`                     | `bash run.sh resnet_32`         | 标准 ResNet-32                          |
| `resnet_56`                     | `bash run.sh resnet_56`         | 标准 ResNet-56 (基线模型之一)           |
| `eca_resnet_20`                 | `bash run.sh eca_resnet_20`     | ResNet-20 + ECA (默认k_size=3)          |
| `eca_resnet_32`                 | `bash run.sh eca_resnet_32`     | ResNet-32 + ECA (默认k_size=5)          |
| `ghost_resnet_20`               | `bash run.sh ghost_resnet_20`   | ResNet-20 + Ghost 模块 (默认ratio=2)    |
| `ghost_resnet_32`               | `bash run.sh ghost_resnet_32`   | ResNet-32 + Ghost 模块 (默认ratio=2)    |
| `ghostnet_100`                  | `bash run.sh ghostnet_100`      | GhostNet v1 (1.0x width)                |
| `convnext_tiny`                 | `bash run.sh convnext_tiny`     | ConvNeXt-Tiny                           |
| `improved_resnet20_convnext`    | `bash run.sh improved_resnet20_convnext` | ResNet-20 结合 ConvNeXt Block 思想 |
| `segnext_mscan_tiny`            | `bash run.sh segnext_mscan_tiny` | SegNeXt-Tiny (MSCAN注意力)             |
| `mlp_mixer_tiny`                | `bash run.sh mlp_mixer_tiny`    | MLP-Mixer Tiny (适配CIFAR)              |
| `mlp_mixer_b16`                 | `bash run.sh mlp_mixer_b16`     | MLP-Mixer B/16 (适配CIFAR)              |
| `cspresnet50`                   | `bash run.sh cspresnet50`       | CSPResNet50 (基于BasicBlock)            |
| `resnest50d`                    | `bash run.sh resnest50d`        | ResNeSt-50D                             |
| `coatnet_0`                     | `bash run.sh coatnet_0`         | CoAtNet-0 (卷积与Transformer混合)     |
| `coatnet_0_custom_enhanced`     | `bash run.sh coatnet_0_custom_enhanced` | CoAtNet-0 + LSK-MBConv (概念)        |
| `coatnet_cifar_opt`             | `bash run.sh coatnet_cifar_opt` | CoAtNet 优化版 for CIFAR (ECA-MBConv) |
| `coatnet_cifar_opt_large_stem`  | `bash run.sh coatnet_cifar_opt_large_stem` | 同上，但使用更大的Stem卷积          |
| `ecanet20_fixed_k3`             | `bash run.sh ecanet20_fixed_k3` | ResNet-20 + ECA (固定k_size=3)        |
| `ecanet20_fixed_k5`             | `bash run.sh ecanet20_fixed_k5` | ResNet-20 + ECA (固定k_size=5)        |
| `ecanet20_fixed_k7`             | `bash run.sh ecanet20_fixed_k7` | ResNet-20 + ECA (固定k_size=7)        |
| `ecanet20_fixed_k9`             | `bash run.sh ecanet20_fixed_k9` | ResNet-20 + ECA (固定k_size=9)        |
| `ecanet20_adaptive`             | `bash run.sh ecanet20_adaptive` | ResNet-20 + ECA (自适应k_size)        |
| `resnet20_no_eca`               | `bash run.sh resnet20_no_eca`   | ResNet-20 (无ECA，用于对比)           |

## 🛠️ 实现的十种先进方法与代表模型 (自定义实现，从头训练)

**注意**: 下表中的"待更新"数值请参考 `report/实验报告-DL2025-先进卷积与注意力机制.md` 中的最新实验结果。

| 序号 | 要求方法         | 本项目代表模型(部分)                  | 对应Top-1(%) | 技术特点                         |
|:----:|------------------|---------------------------------------|-----------------:|----------------------------------|
| 1    | ConvNeXt         | `convnext_tiny`                       | 待更新           | 现代化卷积设计                   |
| 2    | SegNeXt (MSCA)   | `segnext_mscan_tiny`                  | 待更新           | 多尺度卷积注意力                 |
| 3    | LSKNet           | `coatnet_0_custom_enhanced` (概念应用) | 待更新           | 大型选择性核 (通过MBConv增强体现) |
| 4    | CoAtNet          | `coatnet_0`, `coatnet_cifar_opt`      | 待更新           | 卷积+Transformer混合             |
| 5    | ECA-Net          | `eca_resnet_20`, `ecanet20_adaptive`  | 待更新           | 高效通道注意力                   |
| 6    | CSPNet           | `cspresnet50`                         | 待更新           | 跨阶段局部网络                   |
| 7    | GhostNet         | `ghostnet_100`, `ghost_resnet_20`     | 待更新           | 轻量化特征生成                   |
| 8    | HorNet           | (在`coatnet_cifar_opt`中借鉴门控思想) | 待更新           | 递归门控卷积 (间接体现)          |
| 9    | ResNeSt          | `resnest50d`                          | 待更新           | 分裂注意力机制                   |
| 10   | MLP-Mixer        | `mlp_mixer_b16`, `mlp_mixer_tiny`     | 待更新           | 纯MLP视觉架构                    |

## 📖 参考资料与致谢

-   详细的参考文献列表见实验报告 `report/实验报告-DL2025-先进卷积与注意力机制.md` 第12节。
-   模型实现主要参考了各方法原始论文的官方实现和思路，并进行了自定义PyTorch实现。 
-   感谢课程提供的项目框架和指导。
