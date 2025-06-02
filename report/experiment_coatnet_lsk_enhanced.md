# CoAtNet LSK Enhanced: 创新点实验报告

## 1. 引言与目标

CoAtNet 通过结合卷积的局部归纳偏置和 Transformer 的全局上下文建模能力，在多个视觉任务上取得了优异的性能。本项目旨在探索一种针对 CoAtNet 的创新性改进，通过引入 LSK (Large Separable Kernel) Attention 机制来增强其 MBConv 模块的特征提取能力，期望在 CIFAR-100 数据集上获得性能提升或探索新的模型设计思路。

LSKNet 的核心思想是利用大核可分离卷积动态地选择不同感受野的特征，并通过注意力机制进行加权融合。我们将此思想应用于 CoAtNet 中的 MBConv 模块，命名为 `MBConvBlock_enhanced`，并构建了新的模型 `coatnet_0_custom_enhanced`。

本报告将详细记录该创新点的设计、实验设置、多次迭代的实验过程与结果，并对观察到的现象进行分析。

## 2. 创新点详述

### 2.1 LSK (Large Separable Kernel) Attention 机制回顾

LSK Attention 机制源于 LSKNet，其主要组件包括：

*   **多分支大核可分离卷积**: 并行使用多个不同核大小的深度可分离卷积（例如 3x3, 5x5, 7x7）来捕获不同尺度的上下文信息。
*   **空间门控单元 (Spatial Gate Unit)**: 对每个分支的输出进行处理，生成注意力权重。这通常包括一个 1x1 卷积、归一化和激活函数。
*   **加权融合**: 将多分支的输出根据注意力权重进行加权求和，得到最终的增强特征。

### 2.2 MBConvBlock_enhanced 设计

我们将 LSK Attention 的思想融入到标准的 MBConv 模块中，主要修改其深度卷积 (DWConv) 部分：

1.  **并行 DWConv 分支**: 将原 MBConv 中的单个 DWConv 替换为多个并行的 DWConv 分支，每个分支使用 `lsk_kernel_sizes` 中指定的不同核大小。
2.  **LSK Attention 模块**: 在并行 DWConv 分支之后，引入 `LSKAttention` 模块。该模块首先通过全局平均池化和几个线性层来学习每个 DWConv 分支的重要性权重。这些权重随后用于对各个分支的输出进行加权求和。
3.  **结构**:
    ```
    Input
      |
    PW Conv (expansion)
      |
    +----------------------+
    | DW Conv (kernel_1)   |------+
    | DW Conv (kernel_2)   |---- Fused by LSKAttention --> Output
    | ...                  |------+
    | DW Conv (kernel_n)   |
    +----------------------+
      |
    SE Layer (optional, original from MBConv)
      |
    PW Conv (projection)
      |
    Output (with residual connection if applicable)
    ```

### 2.3 CoAtNetCustom_enhanced 架构

新的 `coatnet_0_custom_enhanced` 模型在整体架构上与基础的 `coatnet_0_custom` 保持一致，包括各个阶段 (s0-s4) 的块数量、通道数、Transformer 层的配置等。主要的区别在于 MBConv 阶段（通常是 s0, s1）会使用我们设计的 `MBConvBlock_enhanced` 替换原始的 `MBConvBlock`。

## 3. 实验设置

*   **数据集**: CIFAR-100 (32x32 图像，100个类别)
*   **基础模型参数**:
    *   `coatnet_0_custom` 的标准配置 (块数、通道数等与CoAtNet-0对齐)。
*   **LSK 特定参数 (在 `MBConvBlock_enhanced` 中)**:
    *   `lsk_kernel_sizes`: 默认为 `[3, 5, 7]`
    *   `lsk_reduction_ratio`: 默认为 `8` (用于 `LSKAttention` 内部线性层的通道缩放)
    *   `se_ratio_in_mbconv`: 默认为 `0.25` (MBConv内部SE模块的压缩比例)
*   **训练超参数**:
    *   优化器: AdamW
    *   学习率调度器: Cosine Annealing with Warmup (10 epochs)
    *   权重衰减: 0.05
    *   批大小 (每GPU): 64 (目标值，早期实验可能未使用此值)
    *   总批大小 (8 GPU): 512 (目标值)
    *   训练轮数: 300
    *   标签平滑: 0.1
    *   Mixup: True (alpha=0.2)

## 4. 实验过程与结果

### 4.1 初步尝试 (LR: 1e-3，批大小: 128/GPU，LSK加权方式可能未修正)

*   **现象**: 训练在约第 27 个 epoch 达到约 50.47% 的评估准确率后，损失急剧增加，准确率骤降至 1% 左右，发生训练发散。
*   **日志片段**:
    ```
    2025-06-02 13:06:45,616 - INFO - Epoch 27 评估完成 - 损失: 0.2979, 准确率: 50.47%
    ...
    2025-06-02 13:06:50,939 - INFO - Epoch 28 训练完成 - 损失: 1403.6653, 准确率: 15.18%
    2025-06-02 13:06:51,508 - INFO - Epoch 28 评估完成 - 损失: 1111.4185, 准确率: 1.00%
    ```
*   **分析**: 初始学习率 `1e-3` (乘以 `world_size` 后可能更大) 对于这个包含新注意力机制的模型可能过高，导致训练不稳定。批大小也未使用预期的64。

### 4.2 调整学习率 (LR: 2e-4，批大小: 128/GPU，LSK加权方式可能未修正)

*   **调整**: 将 `src/utils.py` 中 `'Hybrid_Attention_CNN'` 类别的学习率从 `1e-3` 修改为 `2e-4`。
*   **现象**: 根据用户反馈，训练准确率提升缓慢，并在约30%后再次出现不稳定或发散迹象。
*   **分析**: 降低学习率后早期有所改善，但模型依然不稳定。推测 LSK Attention 模块内部的加权求和方式可能存在问题。

### 4.3 修正 LSK Attention 加权方式 (LR: 2e-4，批大小: 128/GPU)

*   **调整**: 修改 `src/model.py` 中 `MBConvBlock_enhanced` 的 `forward` 方法，确保 LSK DW Conv 输出的加权求和使用 `torch.stack` 和正确的广播机制。
*   **遇到的问题**:
    *   `torch.distributed.DistNetworkError: ... EADDRINUSE`: 端口 `29500` 被占用。通过修改 `run.sh` 中 `torchrun` 的 `--master_port` 为 `29501` 解决。
    *   `NameError: name 'fields' is not defined`: `src/trainer.py` 中未从 `dataclasses` 导入 `fields`。通过添加导入解决。
*   **当前训练进展 (2025-06-02 13:08:56 开始的运行)**:
    *   学习率正确应用 `2e-4` (经过 `world_size` 和 warmup 调整)。
    *   批大小仍为 `128/GPU` (未按预期使用 `64/GPU`)。
    *   **初步结果**:
        ```
        2025-06-02 13:08:56,807 - INFO -   学习率 = 0.0
        2025-06-02 13:09:04,500 - INFO - Epoch 2 步骤 [0/48] 损失: 4.6721 准确率: 0.00% 学习率: 0.000160
        ...
        2025-06-02 13:09:33,599 - INFO - Epoch 6 评估完成 - 损失: 0.4072, 准确率: 27.29%
        2025-06-02 13:09:34,782 - INFO - Epoch 7 步骤 [0/48] 损失: 3.7868 准确率: 15.62% 学习率: 0.000960
        ```
        (日志仍在更新中，需要等待更多epochs的结果来判断稳定性)

## 5. 分析与讨论 (待补充)

*   LSK 模块的有效性。
*   不同超参数（尤其是学习率、LSK核大小）对模型性能和稳定性的影响。
*   与标准 CoAtNet-0 的性能对比。
*   训练过程中遇到的挑战及解决方案。

## 6. 结论与未来工作 (待补充)

*   总结本次创新尝试的成果和教训。
*   可能的改进方向：
    *   进一步细致调整 LSK 相关超参数。
    *   尝试不同的 LSK Attention 变体。
    *   将 LSK 模块应用于 CoAtNet 的不同阶段或不同类型的块。
    *   验证批大小问题是否已解决，并使用正确的批大小重新训练。

---
**(此报告将随实验进展持续更新)** 