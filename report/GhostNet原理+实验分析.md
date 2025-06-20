# （一）方法介绍部分

### 原理图：
```
              +----------------+
输入 ------>   | 原始卷积核       | ---> 本征特征图
              +----------------+
                       |
              +------------------------+
              | 廉价的深度可分离卷积      | ---> “鬼影”特征图
              +------------------------+
                       |
       本征特征图 + “鬼影”特征图 --> 拼接 + 裁剪 --> 最终输出
```

### （1）背景

在CNN中，普通卷积层通常需要大量计算资源和参数。研究发现，CNN 中产生的特征图之间存在较强的冗余性——许多特征图内容高度相似，可以视为彼此的“变形”或“副本”。

基于此，GhostNet 提出了一种新颖的轻量化结构设计思路，即通过生成“本征特征图”和利用廉价操作生成“鬼影特征图”，显著减少计算成本，同时保持模型的表达能力。

### （2） 原理

#### （2.1）Ghost模块组成

Ghost 模块将普通卷积拆分为两部分：

- **主卷积部分**  
  使用少量的标准卷积核生成若干本征特征图，负责捕捉核心语义信息。

- **廉价操作部分**  
  对主卷积输出的本征特征图进行轻量级线性操作（如深度可分离卷积），快速生成大量“鬼影”特征图，用以补充丰富的通道表达。

输出特征图即为本征特征图与鬼影特征图拼接裁剪后的结果，数量与普通卷积相同。

#### （2.2）计算量分析

假设输入特征图大小为 \(H \times W\)，输入通道数为 \(C\)，目标输出通道数为 \(n\)，卷积核大小为 \(k \times k\)，Ghost 模块中定义的扩展比例为 \(s\)，则：

- **普通卷积层的计算量（FLOPs）约为：**

\[
\text{FLOPs}_{\text{conv}} = H \times W \times C \times n \times k \times k
\]

因为每个输出通道都需要与所有输入通道做卷积计算。

- **Ghost 模块的计算量由两部分组成：**

1. **主卷积部分计算量：**

\[
\text{FLOPs}_{\text{primary}} = H \times W \times C \times m \times k \times k
\]

其中，\( m = \frac{n}{s} \) 是主卷积输出的本征特征图通道数。

2. **廉价操作部分计算量：**

廉价操作采用深度可分离卷积等线性操作，针对每个通道独立卷积，核大小为 \( d \times d \)，步长为1，通道数为 \( m \)，计算量为：

\[
\text{FLOPs}_{\text{cheap}} = H \times W \times m \times d \times d \times (s - 1)
\]

这里，\( (s - 1) \) 表示每个本征特征图生成的鬼影特征图数目。

- **Ghost模块总计算量：**

\[
\text{FLOPs}_{\text{Ghost}} = \text{FLOPs}_{\text{primary}} + \text{FLOPs}_{\text{cheap}} = H \times W \times C \times \frac{n}{s} \times k \times k + H \times W \times \frac{n}{s} \times d \times d \times (s - 1)
\]

当 \( s \ll C \) 且 \( d \approx k \) 时，计算量近似减少了 \( s \) 倍，相比于普通卷积显著降低了计算成本。

---

这种分解大幅降低了计算复杂度与参数量，使得 Ghost 模块在保持表达能力的同时具备高效的计算性能，适合移动端等资源受限环境的深度网络设计。


### （3）优点


- **计算效率提升**：主卷积核数量减少，深度可分离卷积等廉价操作计算量低，整体计算量降低约 \( s \) 倍。
- **参数量减少**：减少了大量标准卷积参数。
- **特征表达充分**：利用廉价操作生成的鬼影特征能有效补充冗余信息，不影响性能。
- **可插拔模块**：Ghost 模块可替换任意卷积层，易于集成到各种网络架构。


# （二）实验部分

- 数据集：CIFAR-100
- 指标：准确率、参数量、FLOPs、推理时间


### （1）消融实验

**实验设计**： 
    - baseline: ResNet20
    - Ghost_ResNet20 （ratio=2）
    - GhostCopy_ResNet20 （ratio=2）

注：GhostCopy_ResNet20将Ghost的“廉价操作”的深度可分离卷积方法改为直接复制，用于验证深度可分离卷积方法能够保持结果的准确率。

**实验结果** :


**实验结果分析**：



### （2）普通实验

**实验设计**:

- baseline: ResNet20
- GhostNet ratio = 2
- GhostNet ratio = 3
- GhostNet ratio = 4

**实验结果**:

**实验结果分析**：

