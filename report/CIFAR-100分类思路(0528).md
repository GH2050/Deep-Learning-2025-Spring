# **DL-2025 PyTorch CIFAR-100 分类项目：基于ResNet的先进深度学习架构探索与对比分析**

**摘要**

本项目旨在使用PyTorch框架在CIFAR-100图像分类数据集上进行深入研究。以一个精简版的残差网络（ResNet）作为基础模型，核心任务是探索、实现并对比分析至少三种选定的先进深度学习网络架构或注意力机制对基线模型性能的提升效果。这些方法包括但不限于ConvNeXt的设计原则、高效通道注意力（ECA-Net）以及GhostNet等。

**技术栈要求**：项目实现严格遵循以下库的使用规范：
- **PyTorch**：作为主要深度学习框架，负责模型构建、训练和推理
- **datasets**：用于高效的数据集加载、预处理和批处理管理
- **accelerate**：用于训练过程的加速优化和分布式训练支持
- **huggingface_hub**：用于模型版本控制、实验管理和资源共享
- **transformers**：仅使用基础工具组件（如优化器、学习率调度器、配置管理等），不使用任何预训练模型或架构

项目预期产出包括一份详尽的实验报告和相应的演示文稿（PPT），内容涵盖所用方法的理论介绍、具体改进方案的实施细节、全面的实验结果对比、必要的消融研究以验证各模块的有效性，以及对未来工作的展望。本项目的核心挑战在于如何有效地设计和训练这些复杂的深度学习模型，并对模型的复杂性、计算开销与分类性能进行权衡。

**1\. 引言**

CIFAR-100数据集因其类别多样性和图像的细粒度差异，一直是计算机视觉领域中一项具有挑战性的图像分类基准任务 1。它不仅能够有效评估模型对复杂视觉模式的学习能力，也常被用作检验新型网络架构和学习算法有效性的试验场。随着深度学习技术的发展，各种先进的网络结构和注意力机制不断涌现，它们在大型数据集上取得了显著的成就。

本研究的动机在于，系统性地探索和评估多种前沿深度学习技术在改进经典ResNet架构方面的潜力。ResNet以其简洁有效的残差学习机制，成为了许多计算机视觉任务的基石网络 3。然而，在面对如CIFAR-100这类更具挑战性的数据集时，基础的ResNet模型仍有提升空间。通过集成和对比ConvNeXt的设计理念、ECA-Net的高效通道注意力机制、GhostNet的轻量化模块等先进方法，本项目旨在揭示这些技术在提升模型分类准确率、优化参数效率以及降低计算复杂度方面的具体贡献。

本报告将遵循以下结构组织：第二部分将回顾CIFAR-100数据集的特性、ResNet的基础架构以及本项目选定的几种先进深度学习方法的背景和核心思想。第三部分将详细描述用于CIFAR-100分类的简化版ResNet基线模型的实现细节。第四部分将阐述如何将选定的先进技术模块化实现并集成到ResNet基线模型中。第五部分将介绍实验设置，包括数据预处理、训练环境、超参数配置及评估指标。第六部分将呈现并分析各模型在CIFAR-100数据集上的实验结果，包括基线模型性能、改进模型性能对比以及关键的消融实验结果。第七部分（可选）将探讨任何在项目过程中提出的创新性方法及其验证。第八部分将对整个研究工作进行总结，并展望未来的研究方向。第九部分将明确列出团队各成员在项目中的具体贡献。最后，第十和第十一部分分别为参考文献和可选的附录。

**2\. 背景与相关工作**

**2.1 CIFAR-100 数据集**

CIFAR-100 数据集是计算机视觉领域广泛使用的图像分类基准之一 1。该数据集包含100个细粒度类别，这100个类别又被组织成20个粗粒度的超类。每个细粒度类别含有600张32x32像素的彩色图像，其中500张用作训练，100张用作测试，总计训练集包含50000张图像，测试集包含10000张图像 1。图像的小尺寸和类别的多样性为模型学习带来了挑战，要求模型能够捕捉到细微的区分性特征。

在数据预处理方面，通常采用标准化来加速模型收敛并提高稳定性 6。具体而言，图像的每个通道会根据训练集的均值和标准差进行归一化。常用的CIFAR-100均值和标准差（例如，通过torchvision计算得到）为：均值 (0.5071, 0.4867, 0.4408) 和标准差 (0.2675, 0.2565, 0.2761)。此外，数据增强技术对于提升模型的泛化能力至关重要。常见的增强方法包括随机裁剪（例如，在图像四周填充4个像素后进行32x32的随机裁剪）和随机水平翻转 7。这些技术增加了训练数据的多样性，有助于模型学习到更鲁棒的特征表示。

**2.2 ResNet (残差网络)**

残差网络（ResNet）由He等人提出，其核心思想是通过引入残差学习单元和快捷连接（skip connections）来解决深度神经网络训练过程中的梯度消失和网络退化问题 3。传统的深度网络在层数增加时，性能往往先饱和然后迅速下降，而ResNet通过让网络学习残差映射F(x)=H(x)−x（其中H(x)是期望的底层映射，x是输入），使得优化更深层的网络成为可能。

针对CIFAR-100这样的小尺寸图像数据集，通常会采用简化版的ResNet架构。与ImageNet上使用的ResNet相比，主要调整包括：

1. **初始卷积层 (Stem)**：对于32x32的输入图像，标准的ResNet初始7x7卷积层和随后的最大池化层会过度减小特征图尺寸。因此，CIFAR版本的ResNet通常将初始卷积层修改为较小的卷积核（如3x3），步长为1，填充为1，并且可能移除或减小初始池化层的尺寸，以保留更多的空间信息 11。  
2. **残差块 (Residual Blocks)**：根据网络深度，可以选择使用BasicBlock（主要用于ResNet18、ResNet34等较浅网络）或Bottleneck结构（用于ResNet50及更深网络）。BasicBlock由两个3x3卷积层构成，而Bottleneck结构则使用1x1、3x3、1x1的卷积层序列，以在增加网络深度的同时控制参数量和计算量 3。对于本项目的精简版ResNet，采用BasicBlock是常见的选择。  
3. **阶段(Stage)数量和通道数**：为CIFAR数据集设计的ResNet（如ResNet20、ResNet32、ResNet56）通常包含3个主要阶段（stage）。每个阶段由多个残差块堆叠而成。通道数量通常从一个较小的值开始（例如16），并在每个阶段之后翻倍（例如16 \-\> 32 \-\> 64）14。下采样操作通常在每个阶段的第一个残差块中通过设置卷积步长为2来实现。  
4. **全连接层 (Fully Connected Layer)**：在最后一个残差阶段之后，通常会接一个全局平均池化层（Global Average Pooling），然后是一个全连接层，其输出单元数量与目标类别数（CIFAR-100为100）相匹配 3。

这些调整使得ResNet架构能够更有效地处理小尺寸图像，并在CIFAR等数据集上取得优异的分类性能。

**2.3 先进深度学习方法概述**

本项目将从以下列表中选择至少三种方法，在精简版ResNet的基础上进行实现与对比。

* ConvNeXt:  
  ConvNeXt旨在通过借鉴Vision Transformer（尤其是Swin Transformer）的设计原则来"现代化"传统的卷积神经网络（CNN）17。其核心思想并非提出全新的模块，而是通过一系列精心设计的宏观和微观架构调整，系统性地提升标准ResNet的性能。关键的改进机制包括：  
  1. **宏观设计调整**：  
     * *改变阶段计算比例*：调整ResNet中不同阶段（stage）的计算量分布，例如从ResNet-50的(3, 4, 6, 3)块分布调整为更接近Swin-T的(3, 3, 9, 3)分布，以优化FLOPs与精度的平衡 19。  
     * *Patchify Stem*：将ResNet初始的7x7卷积和最大池化层替换为一个更简单的"patchify"层，通常是一个大卷积核（如4x4）、大步长（如4）的卷积操作，直接对输入图像进行分块和下采样，类似于ViT的patch embedding过程 19。  
  2. **ResNeXt化**：  
     * *深度可分离卷积 (Depthwise Convolution)*：采用深度可分离卷积来替代标准卷积，以分离空间特征提取和通道特征混合，这是MobileNet等高效网络的核心组件，也与Transformer中自注意力的加权求和操作在概念上相似 17。  
     * *增加网络宽度*：配合深度卷积的使用，适当增加网络的宽度（通道数），以弥补深度卷积可能带来的表达能力下降，并匹配Swin Transformer的通道数设计 19。  
  3. **倒置瓶颈 (Inverted Bottleneck)**：采用类似于MobileNetV2和Transformer MLP块中的倒置瓶颈结构，即先通过1x1卷积扩大通道数，然后进行深度卷积，最后再通过1x1卷积缩小通道数。这种设计可以在保持甚至减少FLOPs的同时提升性能 17。  
  4. **大卷积核尺寸 (Large Kernel Sizes)**：增加深度卷积层的卷积核尺寸（例如从3x3增加到7x7甚至更大），以扩大感受野，模仿ViT中自注意力的全局信息捕获能力。同时，将深度卷积层在块内的位置上移，使其先于1x1卷积层，确保复杂操作（大核卷积）在较少通道上进行 17。  
  5. **微观设计调整**：  
     * *激活函数替换*：将常用的ReLU激活函数替换为高斯误差线性单元（GELU），GELU在许多Transformer模型中表现更优 17。  
     * *减少激活函数和归一化层数量*：模仿Transformer块的设计，减少残差块中激活函数和归一化层的使用频率，例如只在两个1x1卷积之间保留一个激活函数，并减少BatchNorm (BN) 层的数量 19。  
     * *BatchNorm替换为LayerNorm*：将BatchNorm (BN) 替换为Layer Normalization (LN)。虽然BN对CNN至关重要，但在ConvNeXt的整体设计下，LN也能取得良好效果，并且更接近Transformer的设计 17。  
     * *分离的下采样层*：将空间下采样操作从残差块内部移出，采用独立的2x2、步长为2的卷积层在阶段之间进行下采样，类似于Swin Transformer 19。

ConvNeXt通过这些系统性的改进，使得纯卷积网络在性能上能够与先进的Transformer模型相媲美，同时保持了CNN的效率和简洁性 18。

* ECA-Net (Efficient Channel Attention):  
  ECA-Net 是一种高效的通道注意力机制，旨在以极小的参数量和计算开销提升深度卷积网络的性能 21。它通过分析Squeeze-and-Excitation (SE) 模块的不足，指出避免降维对于学习有效的通道注意力至关重要，并且适当的跨通道交互可以在保持性能的同时显著降低模型复杂度 21。其关键机制包括：  
  1. **无降维的全局平均池化 (GAP)**：与SE模块类似，ECA首先对输入特征图进行通道维度的全局平均池化，以聚合每个通道的空间信息。但与SE不同的是，ECA在此步骤后不进行通道降维 21。  
  2. **局部跨通道交互 (Local Cross-Channel Interaction) 与一维卷积**：ECA模块通过一个快速的一维卷积（Conv1D）来捕获局部跨通道的交互信息。这意味着每个通道的注意力权重是通过考虑其自身以及其k个邻近通道的信息来生成的。这个一维卷积的权重在所有通道间共享，从而大大减少了参数数量 21。  
  3. **自适应核大小 k (Adaptive Kernel Size k)**：一维卷积的核大小k代表了局部跨通道交互的覆盖范围。为了避免手动调整k值，ECA-Net提出了一种自适应方法，通过通道维度C的函数来确定k。具体计算公式为 k=ψ(C)=∣γlog2​(C)​+γb​∣odd​，其中∣t∣odd​表示离t最近的奇数，γ 和 b 是超参数，论文中通常设为2和1 21。这种自适应机制使得ECA模块能够根据不同层级的通道数动态调整交互范围。  
  4. **Sigmoid激活**：一维卷积的输出经过Sigmoid激活函数后，得到每个通道的注意力权重，这些权重随后与原始特征图进行逐通道相乘，以实现特征重标定 21。

ECA模块因其轻量级和有效性，可以方便地集成到现有的CNN架构中（如ResNet、MobileNetV2），在图像分类、目标检测和实例分割等任务上均取得了性能提升 21。

* GhostNet:  
  GhostNet 提出了一种新颖的Ghost模块，旨在通过更少的参数和计算量（即"廉价操作"）生成更多的特征图，从而构建高效的神经网络架构 27。其核心思想是，深度神经网络输出的特征图中存在大量冗余，许多特征图之间高度相似，如同彼此的"幽灵"。GhostNet利用这一特性来减少计算需求。关键机制包括：  
  1. **Ghost模块 (Ghost Module)**：一个标准的卷积层被分为两部分 28：  
     * *主要卷积 (Primary Convolution)*：首先，使用少量标准卷积（例如，标准1x1卷积）来生成一部分"固有"特征图（intrinsic feature maps）。这部分特征图的数量远少于最终期望的输出特征图数量。  
     * *廉价线性操作 (Cheap Linear Operations)*：然后，对这些固有的特征图应用一系列计算成本较低的线性变换（如3x3或5x5的深度卷积、分组卷积，甚至是仿射变换或小波变换的变体），以生成更多的"幽灵"特征图（ghost feature maps）。这些幽灵特征图旨在从固有特征中发掘和增强信息。最后，将固有特征图和幽灵特征图拼接（concatenate）起来，形成最终的输出特征图。  
  2. **Ghost Bottleneck**: Ghost模块可以作为即插即用的组件来升级现有的卷积网络。为了构建完整的GhostNet，论文设计了Ghost Bottleneck结构，它类似于ResNet中的残差块。一个Ghost Bottleneck通常由两个堆叠的Ghost模块组成 28：  
     * 第一个Ghost模块用于扩展通道数量（expansion layer）。  
     * 第二个Ghost模块用于减少通道数量，以匹配快捷连接（shortcut path）的维度。  
     * 与ResNet类似，Ghost Bottleneck也包含快捷连接。当输入和输出通道数不同或需要下采样时，快捷连接路径上可能包含额外的卷积操作（如带步长的深度卷积）。  
  3. **构建GhostNet**: 整个GhostNet架构由一个初始的标准卷积层和一系列堆叠的Ghost Bottleneck构成，最后通过全局平均池化和全连接层进行分类。SE注意力模块也可以选择性地应用于某些Ghost Bottleneck中以进一步提升性能 28。

GhostNet通过这种方式，能够在保持与MobileNetV3等高效模型相当的计算成本（FLOPs）的情况下，实现更高的识别精度，尤其适用于移动设备和嵌入式系统 28。

* CSPNet (Cross Stage Partial Network):  
  CSPNet 是一种旨在增强CNN学习能力同时减少计算瓶颈的网络架构设计策略 31。其核心思想是将每一阶段（stage）的特征图从通道维度分割成两部分，一部分通过一个局部网络块（例如由多个残差块或DenseNet块组成），另一部分则通过一个较短的路径（通常是直接连接或经过少量变换）直接传递到该阶段的末尾，然后两部分特征图进行合并（通常是拼接）并进入下一阶段 33。这种设计有以下几个关键机制和优势：  
  1. **特征图分割与合并 (Feature Map Partitioning and Merging)**：在每个阶段的开始，输入特征图被分成两部分。一部分（例如，一半通道）输入到该阶段的主要计算路径（如一系列ResNet的Bottleneck块），另一部分（另一半通道）则绕过这些复杂计算。在该阶段结束时，这两部分特征图被合并（通常是concatenate），然后可能经过一个过渡层（transition layer）再送入下一阶段 32。  
  2. **跨阶段策略 (Cross-Stage Strategy)**：这种"跨阶段"的部分连接设计使得梯度流可以沿着不同的网络路径传播，从而产生更丰富的梯度组合，减少了梯度信息重复的问题，增强了网络的学习能力 32。  
  3. **减少计算量与内存占用**：由于只有一部分特征图参与了阶段内的主要复杂运算，CSPNet能够显著减少计算量（FLOPs）和内存占用，同时降低了计算瓶颈，提高了推理速度 32。  
  4. **通用性**：CSPNet的设计思想可以应用于多种主流CNN骨干网络，如ResNet、ResNeXt和DenseNet，形成CSPResNet、CSPResNeXt和CSPDenseNet等变体 32。例如，在CSPResNet中，ResNet的每个stage可以被改造成CSP结构，其中一部分特征图经过该stage内的残差块序列，另一部分则直接连接到stage的输出。

CSPNet通过优化梯度流、减少冗余计算和内存访问，在保持甚至提升模型精度的前提下，显著提高了CNN的效率，使其在目标检测等计算密集型任务中表现出色 33。

* ResNeSt (Split-Attention Networks):  
  ResNeSt 是一种改进的ResNet变体，它在ResNet的瓶颈块（Bottleneck Block）中引入了一种新颖的"Split-Attention"模块，旨在使网络能够跨不同的特征图组（feature-map groups）学习注意力权重 40。其关键机制包括：  
  1. **特征图分组 (Feature-map Grouping)**：首先，输入特征图在通道维度上被分成多个"基数组"（cardinal groups），由超参数基数（cardinality）K控制。接着，每个基数组内部进一步被分成R个"分裂组"（splits），由超参数基数率（radix）R控制。因此，总的特征图组数量为G=K×R。每个分裂组都经过一系列相同的变换（例如1x1卷积后接3x3卷积）41。  
  2. **基数组内的分裂注意力 (Split Attention within Cardinal Groups)**：对于每个基数组，其内部R个分裂组的输出特征图首先通过元素级求和进行融合，得到该基数组的聚合表示。然后，对这个聚合表示进行全局平均池化以获得通道描述符（channel-wise statistics）。这些描述符通过两个全连接层（中间有ReLU激活）和一个后续的softmax（当R\>1时）或sigmoid（当R=1时）函数来生成每个分裂组的注意力权重。这些权重是通道相关的，即每个通道对于不同的分裂组有不同的注意力得分 41。  
  3. **加权融合与拼接**：每个基数组的最终输出是通过其内部各分裂组的特征图根据学习到的注意力权重进行加权求和得到的。之后，所有K个基数组的输出特征图沿着通道维度被拼接（concatenate）起来 41。  
  4. **ResNeSt块 (ResNeSt Block)**：拼接后的特征图再经过一个1x1卷积（用于调整通道数和融合信息），然后与输入通过快捷连接（shortcut connection）相加，形成一个完整的ResNeSt块。如果需要下采样，则快捷连接路径和主路径都会进行相应的步长调整 41。

ResNeSt通过这种方式，使得网络不仅能够像ResNeXt那样利用多分支结构（通过基数组实现），还能在每个分支内部（通过分裂组）动态地学习不同子特征的重要性，从而更有效地捕获跨通道的特征依赖关系和更丰富的上下文信息。这使得ResNeSt在图像分类、目标检测和分割等多种任务上均取得了优于标准ResNet及其它变体的性能 40。

* MLP-Mixer:  
  MLP-Mixer 是一种完全基于多层感知机（MLP）的视觉架构，它不使用卷积操作或自注意力机制，这与主流的CNN和Transformer模型有显著区别 46。其核心思想是通过交替应用两种类型的MLP层来分别混合空间信息和通道信息。关键机制包括：  
  1. **图像分块与线性嵌入 (Patch Embedding)**：输入图像首先被分割成一系列不重叠的图像块（patches），类似于Vision Transformer (ViT) 的做法。每个图像块被线性投影（通过一个共享的嵌入矩阵）成一个固定维度的特征向量（token）。这些tokens形成一个形状为"patches × channels"的二维表作为后续Mixer层的输入 47。  
  2. **Mixer层 (Mixer Layers)**：MLP-Mixer由多个相同的Mixer层堆叠而成。每个Mixer层包含两种MLP块：  
     * **Token-mixing MLP (跨块MLP)**：这类MLP独立地作用于输入表的每一列（即每个通道），混合不同图像块（tokens）之间的空间信息。它包含两个全连接层和一个GELU非线性激活函数。输入首先进行层归一化（Layer Normalization），然后通过MLP，最后通过快捷连接与原始输入相加 47。  
     * **Channel-mixing MLP (块内通道MLP)**：这类MLP独立地作用于输入表的每一行（即每个图像块），混合每个图像块内部不同通道之间的特征信息。其结构与Token-mixing MLP类似，也包含两个全连接层、GELU激活、层归一化和快捷连接 47。  
  3. **分类头 (Classifier Head)**：经过多层Mixer处理后，对所有图像块的特征表示进行全局平均池化，然后通过一个标准的全连接线性分类器得到最终的类别预测 47。

与CNN不同，MLP-Mixer不依赖卷积的局部感受野和权重共享带来的平移不变性（尽管通道混合MLP的参数在所有块间共享，可以看作一种1x1卷积）。与Transformer不同，它不使用自注意力机制来动态地计算不同块之间的关系权重，而是使用固定的MLP权重来混合信息。MLP-Mixer的参数量和计算复杂度与输入图像分辨率和块大小有关，当在大规模数据集上预训练时，MLP-Mixer能够达到与CNN和Transformer相当的性能，且推理成本具有竞争力 47。

**3\. 基础模型实现 (Simplified ResNet for CIFAR-100)**

为了在CIFAR-100数据集上进行有效的比较分析，首先需要构建一个合适的基线模型。本项目采用了一个针对32x32图像尺寸优化的简化版ResNet。

**3.1 架构细节**

该简化ResNet的架构设计如下：

* **输入层**：接收3通道、32x32像素的CIFAR-100图像。  
* **初始卷积层 (conv1)**：为了适应较小的输入尺寸并保留更多初始特征信息，初始卷积层采用一个3x3的卷积核，步长（stride）为1，填充（padding）为1。该层输出16个通道，之后连接批量归一化（BatchNorm2d）和ReLU激活函数 12。与在ImageNet上常用的ResNet不同，这里不使用初始的最大池化层（MaxPool），以避免过早地大幅降低特征图分辨率 12。  
* **残差阶段 (Layers/Stages)**：模型包含三个残差阶段，每个阶段由若干个BasicBlock堆叠而成。BasicBlock是ResNet中用于较浅网络（如ResNet18, ResNet34）的基础残差单元，包含两个3x3卷积层，每个卷积层后接BatchNorm2d和ReLU 3。  
  * **Stage 1**: 输出通道数为16。假设包含 n1​ 个BasicBlock。所有块的步长均为1。  
  * **Stage 2**: 输出通道数为32。假设包含 n2​ 个BasicBlock。该阶段的第一个BasicBlock中的第一个卷积层步长为2，用于将特征图尺寸减半（空间下采样）。其余块步长为1。  
  * **Stage 3**: 输出通道数为64。假设包含 n3​ 个BasicBlock。该阶段的第一个BasicBlock中的第一个卷积层步长也为2，进行第二次空间下采样。其余块步长为1。  
  * 具体的块数量 n1​,n2​,n3​ 可以根据ResNet的典型配置（如ResNet20中通常为(3,3,3)，ResNet32中为(5,5,5)）进行选择，以构建一个"精简版"的ResNet 14。对于本项目，可以考虑使用类似ResNet20的配置，即每个阶段3个BasicBlock。  
* **全局平均池化层 (Global Average Pooling)**：在最后一个残差阶段之后，应用自适应全局平均池化层 (nn.AdaptiveAvgPool2d((1, 1)))，将每个通道的特征图缩减为一个单一值。  
* **全连接层 (Classifier)**：最后，连接一个全连接层（nn.Linear），其输入特征数量为最后一个残差阶段的输出通道数（在此设计中为64），输出特征数量为CIFAR-100的类别数，即100。

**3.2 PyTorch实现要点**

使用PyTorch实现上述简化版ResNet时，主要遵循以下步骤：

**核心库使用规范**：
* **PyTorch (torch)**：
  - torch.nn.Module：用于定义模型结构
  - torch.nn.Conv2d, torch.nn.BatchNorm2d 等：基础网络层
  - torch.optim：优化器实现
  - torch.utils.data.DataLoader：数据加载
* **datasets**：
  - 替代 torchvision.datasets 进行数据集管理
  - 提供更灵活的数据预处理pipeline
* **accelerate**：
  - Accelerator：统一处理设备管理、混合精度训练
  - 简化多GPU训练流程
* **huggingface_hub**：
  - 模型版本管理和实验追踪
  - Repository API用于代码和模型管理
* **transformers（仅基础构件）**：
  - get_scheduler：学习率调度器
  - AdamW优化器：替代torch.optim.AdamW
  - TrainingArguments：训练配置管理
  - 严格禁止使用任何预训练模型或Transformer架构

1. **模型类定义**：创建一个继承自 torch.nn.Module 的类来定义整个ResNet模型。  
2. **层定义**：在类的 \_\_init\_\_ 方法中，实例化所需的各个网络层，包括：  
   * torch.nn.Conv2d 用于卷积层。  
   * torch.nn.BatchNorm2d 用于批量归一化层。  
   * torch.nn.ReLU(inplace=True) 用于ReLU激活函数。  
   * torch.nn.AdaptiveAvgPool2d((1, 1)) 用于全局平均池化。  
   * torch.nn.Linear 用于全连接分类层。  
   * BasicBlock本身也应定义为一个 nn.Module 子类，包含其内部的卷积、BN、ReLU层以及快捷连接逻辑。快捷连接需要处理输入输出通道数不匹配或空间尺寸不匹配的情况（通常通过一个带步长的1x1卷积实现）。  
3. **前向传播逻辑**：在类的 forward 方法中，定义数据从输入到输出的完整计算流程，即如何依次通过初始卷积层、各个残差阶段、全局平均池化层和最终的全连接层。
4. **训练流程**：使用 accelerate.Accelerator 管理训练过程，使用 datasets 管理数据流，使用 transformers 的基础工具进行优化和调度。

模型定义本身不直接依赖于CPU或GPU。在训练和推理时，可以根据可用硬件选择合适的设备（CPU或GPU）。

以下是一个BasicBlock的简化PyTorch实现示例：

Python

import torch.nn as nn  
import torch.nn.functional as F

class BasicBlock(nn.Module):  
    expansion \= 1

    def \_\_init\_\_(self, in\_planes, planes, stride=1):  
        super(BasicBlock, self).\_\_init\_\_()  
        self.conv1 \= nn.Conv2d(in\_planes, planes, kernel\_size=3, stride=stride, padding=1, bias=False)  
        self.bn1 \= nn.BatchNorm2d(planes)  
        self.conv2 \= nn.Conv2d(planes, planes, kernel\_size=3, stride=1, padding=1, bias=False)  
        self.bn2 \= nn.BatchNorm2d(planes)

        self.shortcut \= nn.Sequential()  
        if stride\!= 1 or in\_planes\!= self.expansion \* planes:  
            self.shortcut \= nn.Sequential(  
                nn.Conv2d(in\_planes, self.expansion \* planes, kernel\_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(self.expansion \* planes)  
            )

    def forward(self, x):  
        out \= F.relu(self.bn1(self.conv1(x)))  
        out \= self.bn2(self.conv2(out))  
        out \+= self.shortcut(x)  
        out \= F.relu(out)  
        return out

整个ResNet模型将由这些BasicBlock和上述其他组件构成。

**4\. 改进方法实现与集成**

本章节将详细阐述如何实现选定的先进深度学习方法的核心模块，并将其集成到先前定义的简化版ResNet基线模型中。我们以ECA-Net、GhostNet和ConvNeXt的设计原则为例进行说明。

**4.1 ECA-ResNet: 集成高效通道注意力**

* 4.1.1 ECA模块实现  
  ECA模块的核心是通过不降维的全局平均池化（GAP）和快速一维卷积（Conv1D）来捕获局部跨通道交互信息，并使用Sigmoid函数生成注意力权重 21。一维卷积的核大小 k 可以通过通道数 C 自适应确定： k=∣γlog2​(C)​+γb​∣odd​，其中 γ=2,b=1 是常用设置 21。  
  PyTorch实现ECA模块可以参考官方或可靠的第三方代码库，如 timm 库或原始论文作者提供的实现 54。一个简化的ECA模块PyTorch实现如下：  
  Python  
  import torch  
  import torch.nn as nn  
  import math

  class eca\_layer(nn.Module):  
      """Constructs an ECA module.  
      Args:  
          channel: Number of channels of the input feature map  
          k\_size: Adaptive selection of kernel size  
      """  
      def \_\_init\_\_(self, channel, k\_size=3): \# k\_size can be made adaptive  
          super(eca\_layer, self).\_\_init\_\_()  
          self.avg\_pool \= nn.AdaptiveAvgPool2d(1)  
          self.conv \= nn.Conv1d(1, 1, kernel\_size=k\_size, padding=(k\_size \- 1) // 2, bias=False)  
          self.sigmoid \= nn.Sigmoid()  
          self.channel \= channel \# For adaptive k\_size calculation  
          self.k\_size \= self.\_calculate\_k\_size(channel) \# Calculate adaptive k\_size  
          \# Re-initialize conv layer with adaptive k\_size  
          self.conv \= nn.Conv1d(1, 1, kernel\_size=self.k\_size, padding=(self.k\_size \- 1) // 2, bias=False)

      def \_calculate\_k\_size(self, C, gamma=2, b=1):  
          t \= int(abs((math.log2(C) / gamma) \+ (b / gamma)))  
          k \= t if t % 2 else t \+ 1  
          return k

      def forward(self, x):  
          \# feature descriptor on the global spatial information  
          y \= self.avg\_pool(x)

          \# Two different branches of ECA module  
          y \= self.conv(y.squeeze(-1).transpose(-1, \-2)).transpose(-1, \-2).unsqueeze(-1)

          \# Multi-scale information fusion  
          y \= self.sigmoid(y)

          return x \* y.expand\_as(x)

  在实际使用中，k\_size应根据输入通道数动态计算。  
* 4.1.2 集成到ResNet  
  ECA模块通常被插入到ResNet的残差块（BasicBlock或Bottleneck）中。一种常见的位置是在残差分支的最后一个卷积层之后，但在与快捷连接（shortcut）相加之前 21。对于BasicBlock，ECA模块可以放置在第二个3x3卷积和BN层之后。  
  修改后的BasicBlock (BasicBlockECA) 结构示意图：  
  Input \-\> Conv1 \-\> BN1 \-\> ReLU \-\> Conv2 \-\> BN2 \-\> ECA \-\> \+ Shortcut \-\> ReLU \-\> Output  
* 4.1.3 参数量和计算量考量  
  ECA模块引入的额外参数量非常少，仅为 k 个（由于一维卷积权重共享）。计算量增量也极小，主要是一维卷积和池化操作。相比SE模块，ECA在效率上有明显优势 21。

**4.2 Ghost-ResNet: 集成Ghost模块**

* 4.2.1 Ghost模块实现  
  Ghost模块将标准卷积分为两步：1) 使用少量标准卷积（通常是1x1卷积）生成固有特征图；2) 对固有特征图应用一系列廉价的线性操作（通常是深度卷积）生成幽灵特征图，然后拼接两者 28。  
  PyTorch实现Ghost模块可参考官方或timm库 57。一个Ghost模块的核心逻辑如下：  
  Python  
  import torch  
  import torch.nn as nn

  class GhostModule(nn.Module):  
      def \_\_init\_\_(self, inp, oup, kernel\_size=1, ratio=2, dw\_size=3, stride=1, relu=True):  
          super(GhostModule, self).\_\_init\_\_()  
          self.oup \= oup  
          init\_channels \= math.ceil(oup / ratio)  
          new\_channels \= init\_channels \* (ratio \- 1)

          self.primary\_conv \= nn.Sequential(  
              nn.Conv2d(inp, init\_channels, kernel\_size, stride, kernel\_size//2, bias=False),  
              nn.BatchNorm2d(init\_channels),  
              nn.ReLU(inplace=True) if relu else nn.Sequential(),  
          )

          self.cheap\_operation \= nn.Sequential(  
              nn.Conv2d(init\_channels, new\_channels, dw\_size, 1, dw\_size//2, groups=init\_channels, bias=False),  
              nn.BatchNorm2d(new\_channels),  
              nn.ReLU(inplace=True) if relu else nn.Sequential(),  
          )

      def forward(self, x):  
          x1 \= self.primary\_conv(x)  
          x2 \= self.cheap\_operation(x1)  
          out \= torch.cat(\[x1, x2\], dim=1)  
          return out\[:, :self.oup, :, :\]

* 4.2.2 集成到ResNet  
  要构建Ghost-ResNet，可以将ResNet中的标准卷积层（尤其是BasicBlock或Bottleneck内部的3x3卷积）替换为Ghost模块 28。如果替换BasicBlock中的卷积层，需要注意通道数的变化和快捷连接的匹配。更直接的方法是设计Ghost Bottleneck，它本身就包含了快捷连接，然后用Ghost Bottleneck替换ResNet中的标准Bottleneck或调整BasicBlock以适应Ghost模块。对于精简版ResNet，可以将BasicBlock内的两个3x3卷积都替换或部分替换为GhostModule。  
  修改后的BasicBlock (BasicBlockGhost) 结构示意图 (替换第一个卷积为例)：  
  Input \-\> GhostModule1 \-\> BN1 \-\> ReLU \-\> Conv2 \-\> BN2 \-\> \+ Shortcut \-\> ReLU \-\> Output  
  或者，将两个卷积都替换：  
  Input \-\> GhostModule1 \-\> BN1 \-\> ReLU \-\> GhostModule2 \-\> BN2 \-\> \+ Shortcut \-\> ReLU \-\> Output  
* 4.2.3 参数量和计算量考量  
  Ghost模块通过将昂贵的标准卷积分解为更少通道的标准卷积和廉价的深度卷积，能够显著减少参数量和FLOPs，同时保持有竞争力的性能 28。压缩比和加速比理论上接近于ratio参数。

**4.3 ConvNeXt-inspired ResNet: 应用ConvNeXt设计原则**

* 4.3.1 ConvNeXt核心原则应用  
  将ConvNeXt的设计原则应用于ResNet，不是简单替换一个模块，而是一系列架构上的调整 17。对于精简版ResNet，可以考虑以下关键修改：  
  1. **Patchify Stem**: 将初始的3x3 conv1 替换为一个4x4、步长为4的卷积层（如果输入尺寸允许，对于32x32可能调整为2x2，步长2，或保留3x3，步长1，但后续阶段调整）。  
  2. **倒置瓶颈 (Inverted Bottleneck) 与深度卷积**: 修改BasicBlock。可以将BasicBlock中的两个3x3标准卷积替换为一个倒置瓶颈结构：一个1x1卷积升维，接一个大核（如7x7，或CIFAR-100上可尝试5x5）的深度卷积，再接一个1x1卷积降维。  
  3. **激活函数与归一化**: 在新的块结构中，使用GELU替换ReLU，并尝试使用LayerNorm替换BatchNorm2d，或减少BN层数量。  
  4. **阶段计算比例**: 可以调整每个ResNet阶段的BasicBlock数量，例如从(3,3,3)调整为更接近ConvNeXt-T的(3,3,9,3)中的前三部分（需要适配总层数）。

PyTorch实现这些改动涉及到对BasicBlock乃至整个ResNet类的深度重构。timm库中ConvNeXt的实现可以作为参考 62。一个受ConvNeXt启发的块 (ConvNeXtLikeBlock) 的伪代码结构：Python  
\# ConvNeXtLikeBlock  
\# Input x  
\# dw\_conv \= DepthwiseConv2d(kernel\_size=7, padding=3)  
\# norm \= LayerNorm()  
\# pw\_conv1 \= LinearLayer() \# Pointwise conv (1x1 equivalent)  
\# act \= GELU()  
\# pw\_conv2 \= LinearLayer() \# Pointwise conv (1x1 equivalent)  
\# LayerScale (optional)

\# x\_residual \= x  
\# x \= dw\_conv(x)  
\# x \= norm(x)  
\# x \= pw\_conv1(x)  
\# x \= act(x)  
\# x \= pw\_conv2(x)  
\# x \= layer\_scale(x)  
\# x \= x\_residual \+ x  
\# Output x  
在ResNet中集成时，这个块将替换原有的BasicBlock。

* 4.3.2 集成到ResNet  
  集成ConvNeXt原则意味着对ResNet的整体架构进行现代化改造。这包括修改Stem层，用ConvNeXt风格的块替换原有的BasicBlock，并调整归一化和激活策略。下采样层也可能需要调整为独立的卷积层。  
* 4.3.3 参数量和计算量考量  
  ConvNeXt的设计目标之一是在现代硬件上实现更好的FLOPs/准确率平衡。倒置瓶颈和深度卷积通常能减少FLOPs，但增加网络宽度或卷积核大小则可能增加计算量。LayerNorm相对于BatchNorm在某些情况下计算成本可能更高。精确的参数和FLOPs变化取决于具体的实现细节。

对于每种选择的改进方法，都需要仔细设计其在ResNet中的集成方式，确保与训练环境的兼容性，并为后续的实验对比和消融研究打下基础。

**5\. 实验设置**

为确保实验结果的可靠性和可复现性，本节详细说明实验所用的数据集、数据预处理方法、训练环境、超参数配置以及评估性能的指标。

**5.1 数据集与预处理**

* **数据集**: 本项目采用CIFAR-100数据集 1。该数据集包含100个类别，每类有500张训练图像和100张测试图像，图像尺寸为32x32像素的彩色图像。训练集共50,000张，测试集共10,000张。  
* **数据预处理与增强**:  
  * **标准化 (Normalization)**: 对图像的RGB三个通道进行标准化处理。根据CIFAR-100数据集的统计特性，常用的均值和标准差为：均值 mean \= \[0.5071, 0.4867, 0.4408\]，标准差 std \= \[0.2675, 0.2565, 0.2761\] 8。torchvision.transforms.Normalize(mean, std)将被用于此目的。  
  * **数据增强 (Data Augmentation)**: 为提高模型的泛化能力，对训练集采用以下数据增强技术 7：  
    * 随机裁剪 (Random Crop): 首先对图像进行填充（padding），在图像四周各填充4个像素，使其尺寸变为40x40，然后从中随机裁剪出32x32大小的图像。这通过 torchvision.transforms.RandomCrop(32, padding=4) 实现。  
    * 随机水平翻转 (Random Horizontal Flip): 以0.5的概率对图像进行水平翻转。这通过 torchvision.transforms.RandomHorizontalFlip() 实现。  
    * 转换为张量 (To Tensor): 将PIL图像或NumPy数组转换为PyTorch张量，并将像素值从范围缩放到\[0.0, 1.0\]范围。这通过 torchvision.transforms.ToTensor() 实现。 测试集仅进行转换为张量和标准化处理，不进行随机数据增强。

训练集转换流程: transforms.Compose()测试集转换流程: transforms.Compose()

**5.2 训练环境与超参数**

* **训练环境**: 所有实验均在PyTorch环境下进行，支持CPU和GPU训练。模型设计和训练参数的选择将根据可用硬件资源进行调整。  
* **优化器 (Optimizer)**:  
  * 选用随机梯度下降（SGD）优化器，设置动量（momentum）为0.9，权重衰减（weight decay）为5e-4 9。  
  * 或者，可以考虑使用AdamW优化器，它在Transformer和一些现代CNN训练中表现良好，其权重衰减的处理方式与标准Adam不同 8。若使用AdamW，初始学习率通常设置得较小，例如1e-3或5e-4。  
* **学习率调度 (Learning Rate Schedule)**:  
  * 采用余弦退火学习率调度策略 (torch.optim.lr\_scheduler.CosineAnnealingLR) 8。这是一种平滑降低学习率的方式，有助于模型在训练后期更好地收敛。  
  * 或者，可以采用阶梯式学习率衰减 (torch.optim.lr\_scheduler.StepLR 或 MultiStepLR)，例如在总训练轮数的特定比例（如50%、75%）时将学习率乘以一个衰减因子（如0.1或0.2）71。  
  * 初始学习率：对于SGD，可以从0.1开始尝试 71；对于AdamW，可以从0.001开始尝试 8。  
* **批量大小 (Batch Size)**: 批量大小将根据实际可用内存进行调整，常见的选择有64、128或256。对于GPU训练，可以使用更大的批量大小以提高训练效率。  
* **训练轮数 (Epochs)**: 训练深度模型的轮数通常在100到200个轮次之间。具体轮数可根据实际训练时间和收敛情况进行调整。  
* **损失函数 (Loss Function)**: 对于多类别分类任务CIFAR-100，采用交叉熵损失函数 (torch.nn.CrossEntropyLoss) 9。

**5.3 评估指标**

为了全面评估和比较不同模型的性能，将采用以下指标：

* **Top-1 准确率 (Top-1 Accuracy)**: 模型预测的最可能的类别与真实类别一致的样本比例。这是图像分类任务中最主要的性能衡量标准。  
* **Top-5 准确率 (Top-5 Accuracy)**: 模型预测的概率最高的5个类别中包含真实类别的样本比例。该指标对于具有大量类别的任务（如CIFAR-100）尤为重要，可以反映模型是否能将真实类别排在前列。  
* **模型参数量 (Number of Parameters)**: 模型中所有可训练参数的总数。该指标反映了模型的存储需求和潜在的复杂性。  
* **浮点运算次数 (FLOPs \- Floating Point Operations)**: 模型进行一次前向传播所需的浮点运算次数。该指标是衡量模型计算复杂度的常用标准。可以使用如 thop (PyTorch-OpCounter) 这样的库来计算模型的FLOPs。

这些指标将共同用于评估所实现模型在分类性能、模型大小和计算效率方面的表现。

**6\. 实验结果与分析**

本章节将呈现并分析在CIFAR-100数据集上，基于简化版ResNet及其集成先进技术后的各个模型的实验结果。分析将围绕模型的分类准确率、参数量和计算复杂度（FLOPs）展开。

**6.1 基线模型性能**

首先，对第3节中描述的简化版ResNet（例如，类似ResNet20的结构，包含3个阶段，每阶段3个BasicBlock，通道数分别为16, 32, 64）在CIFAR-100上进行训练和评估。其性能将作为后续改进模型比较的基准。

假设基线模型（Simplified ResNet-20）的实验结果如下（这些数值为示例，实际数值需通过实验获得）：

* Top-1 准确率: 65.8%  
* Top-5 准确率: 88.2%  
* 参数量: 0.27 M  
* FLOPs: 40 M

此基线性能反映了基础的ResNet架构在CIFAR-100上的能力。

**6.2 改进模型性能对比**

接下来，将基线模型与集成了ECA-Net、GhostNet和ConvNeXt启发式改进的ResNet模型进行性能对比。所有模型均在相同的实验设置（如第5节所述）下进行训练和评估。

**表6.1: 不同模型在CIFAR-100上的性能对比**

| 模型 | Top-1 准确率 (%) | Top-5 准确率 (%) | 参数量 (M) | FLOPs (M) | 相对基线Top-1提升 (%) |
| :---- | ----- | ----- | ----- | ----- | ----- |
| Simplified ResNet-20 (基线) | 65.8 | 88.2 | 0.27 | 40 | \- |
| ECA-ResNet-20 | 67.5 | 89.1 | 0.271 | 40.05 | \+1.7 |
| Ghost-ResNet-20 (s=2) | 66.5 | 88.8 | 0.15 | 22 | \+0.7 |
| ConvNeXt-like-ResNet-T\_CIFAR | 69.2 | 90.5 | 0.35 | 55 | \+3.4 |
| *\[其他实现的方法\]* | TBD | TBD | TBD | TBD | TBD |

*(注: 表中数据为假设的示例值，用于说明分析框架，实际数值以实验结果为准。ConvNeXt-like-ResNet-T\_CIFAR 指的是一个根据ConvNeXt Tiny版本原则调整通道数和块数的ResNet变体，参数量和FLOPs会根据具体实现而变化。)*

**结果分析:**

1. **准确率提升**:  
   * 从示例数据看，ConvNeXt启发式改进的ResNet（ConvNeXt-like-ResNet-T\_CIFAR）在Top-1准确率上可能带来最显著的提升（例如，+3.4%）。这得益于其对ResNet架构进行的一系列现代化改造，如倒置瓶颈、大卷积核、GELU激活和LayerNorm等，这些设计已被证明能有效提升模型性能 17。  
   * ECA-ResNet-20也显示出积极的性能提升（例如，+1.7%），表明高效通道注意力机制能够在几乎不增加额外参数和计算量的情况下，有效增强模型对通道特征的辨别能力 21。  
   * Ghost-ResNet-20在准确率上的提升可能相对温和（例如，+0.7%），但其主要优势在于显著降低参数量和FLOPs。  
2. **效率 (参数量/FLOPs与准确率的权衡)**:  
   * Ghost-ResNet-20在效率方面表现突出。如示例所示，它可能以略微的准确率提升为代价，大幅减少了参数量（例如，从0.27M降至0.15M）和FLOPs（例如，从40M降至22M）28。这使其成为资源受限环境下的一个有吸引力的选择。  
   * ECA-ResNet-20在参数量和FLOPs上与基线模型几乎持平，但带来了明确的准确率增益，展现了极高的效率 21。  
   * ConvNeXt-like-ResNet-T\_CIFAR虽然准确率提升最明显，但其参数量和FLOPs也可能相应增加（例如，参数量增至0.35M，FLOPs增至55M）。这需要在具体应用中权衡性能和资源消耗。  
3. 训练时长考量:  
   FLOPs更高的模型（如ConvNeXt-like-ResNet）通常意味着更长的单轮训练时间。GhostNet由于其廉价操作，训练速度可能会相对较快。ECA-Net由于其轻量级特性，对训练时长的影响也应较小。记录并比较各模型的实际训练总时长，对于评估它们的实用性具有重要价值。

**6.3 消融实验**

为了验证所选改进方法中各个关键组件的有效性，将进行一系列消融实验。

* **ECA-Net 消融实验**:  
  * **组件1: 自适应核大小 k vs. 固定 k**。比较使用自适应计算得到的核大小 k 的ECA模块，与使用固定 k 值（例如 k=3,5,7）的ECA模块的性能差异。预期自适应 k 能在不同层（通道数不同）提供更优的性能 21。  
  * **组件2: 1D卷积 vs. 全连接层 (模拟SE)**。可以将ECA中的1D卷积替换为类似SE模块中的两个全连接层（带瓶颈结构或不带），以对比ECA的1D卷积在效率和效果上的优势。  
  * **组件3: ECA模块的有无**。即ECA-ResNet与基线ResNet的直接对比，这已在6.2节中体现。

**表6.2: ECA-Net消融实验结果 (示例)**

| ECA配置 | Top-1 准确率 (%) | 参数量 (M) | FLOPs (M) |
| :---- | ----- | ----- | ----- |
| ResNet-20 (无ECA) | 65.8 | 0.27 | 40 |
| ECA-ResNet-20 (固定 k=3) | 67.2 | 0.2705 | 40.02 |
| ECA-ResNet-20 (固定 k=5) | 67.3 | 0.2708 | 40.03 |
| ECA-ResNet-20 (自适应 k) | 67.5 | \~0.271 | \~40.05 |

分析：从示例数据可以看出，自适应核大小的ECA模块可能提供最佳性能，而固定核大小也能带来改进，但可能不如自适应版本鲁棒。

* **GhostNet 消融实验**:  
  * **组件1: Ghost模块中的廉价操作类型**。如果Ghost模块实现中允许选择不同的廉价操作（如不同核大小的深度卷积），可以比较它们的效果。  
  * **组件2: Ghost模块的比例参数 s (或 ratio)**。比较不同 s 值（例如 s=2,s=3,s=4）对模型压缩率和准确率的影响。理论上，s 越大，参数和计算量越低，但准确率可能会有所下降 28。  
  * **组件3: Ghost模块在BasicBlock中的替换策略**。比较仅替换第一个3x3卷积、仅替换第二个3x3卷积、或同时替换两个3x3卷积的效果。

**表6.3: GhostNet消融实验结果 (示例 \- 改变 s 值)**

| Ghost-ResNet-20 配置 | Top-1 准确率 (%) | 参数量 (M) | FLOPs (M) |
| :---- | ----- | ----- | ----- |
| ResNet-20 (基线) | 65.8 | 0.27 | 40 |
| Ghost-ResNet (s=2) | 66.5 | 0.15 | 22 |
| Ghost-ResNet (s=3) | 65.9 | 0.11 | 16 |
| Ghost-ResNet (s=4) | 65.2 | 0.09 | 13 |

分析：示例数据显示，随着 \<span class="math-inline"\>s\</span\> 值的增加，参数量和FLOPs显著降低，但Top-1准确率也随之下降。需要在效率和性能之间找到平衡点。

* ConvNeXt-inspired ResNet 消融实验:  
  ConvNeXt的改进是多方面的，可以逐项或分组评估其关键设计原则对ResNet性能的影响 19。  
  * **组件1: 倒置瓶颈与深度卷积**。在BasicBlock中引入倒置瓶颈和深度卷积，与标准BasicBlock比较。  
  * **组件2: 大卷积核**。在采用深度卷积的基础上，比较不同卷积核大小（如3x3, 5x5, 7x7）的效果。  
  * **组件3: GELU激活函数**。将ReLU替换为GELU，观察性能变化。  
  * **组件4: LayerNorm**。将BatchNorm2d替换为LayerNorm，观察性能变化（可能需要结合其他ConvNeXt改动）。  
  * **组件5: Patchify Stem**。将ResNet的传统Stem替换为ConvNeXt的Patchify Stem。

**表6.4: ConvNeXt-inspired ResNet消融实验结果 (示例 \- 逐步添加特性)**

| 模型配置 | Top-1 准确率 (%) |
| :---- | ----- |
| ResNet-20 (基线) | 65.8 |
| \+ 倒置瓶颈 & 深度卷积 (3x3核) | 67.0 |
| \+ 倒置瓶颈 & 深度卷积 (7x7核) | 67.8 |
| \+ 倒置瓶颈 & 深度卷积 (7x7核) & GELU | 68.1 |
| \+ 倒置瓶颈 & 深度卷积 (7x7核) & GELU & LayerNorm | 68.5 |
| \+ 全套ConvNeXt微观设计 (含Patchify Stem等) | 69.2 |

分析：示例结果表明，ConvNeXt的各项设计原则逐步引入，均可能对基线ResNet的性能产生积极影响。其中，倒置瓶颈、大卷积核以及更现代的激活/归一化策略是关键贡献点。

通过这些详尽的实验对比和消融研究，可以更清晰地理解每种先进技术在CIFAR-100分类任务上对简化版ResNet的改进效果及其内部机制的有效性，为在资源受限环境下选择和优化模型提供有力的依据。

**7\. 创新性方法 (可选加分项)**

在完成基础要求之上，可以尝试提出并验证一种创新性的方法，旨在进一步提升模型在CIFAR-100分类任务上的性能或效率。

**7.1 创新点详述**

一个可能的创新方向是结合不同先进技术的优势，设计一种混合模块或架构。例如，可以考虑**将GhostNet的轻量化思想与ECA-Net的高效通道注意力机制相结合，构建一个"幽灵高效通道注意力"（GhostECA）模块**。

**GhostECA模块设计思路：**

1. **基础结构**：借鉴GhostNet的思想，将特征图生成分为两部分：一部分通过主要的卷积路径，另一部分通过廉价操作生成。  
2. **主要路径集成ECA**：在主要卷积路径（例如，GhostModule中的primary\_conv之后，或者在整个GhostModule输出之后）嵌入一个ECA模块。这样，主要的、信息量更丰富的特征图可以首先通过ECA进行通道注意力的优化。  
3. **廉价操作部分**：保持GhostNet原有的廉价操作（如深度卷积）来生成"幽灵"特征图。  
4. **特征融合**：将经过ECA优化的主要特征图与生成的幽灵特征图进行拼接。

**预期优势**：

* **效率**：继承GhostNet的特性，通过廉价操作减少计算量和参数量。  
* **性能**：通过ECA模块增强主要特征图的表达能力，弥补GhostNet在特征提取深度上可能存在的不足，同时避免了ECA模块应用于所有（包括冗余的）特征图上可能造成的轻微计算浪费。  
* **硬件友好**：两种机制本身都相对轻量，其组合有望在CPU和GPU上都保持较好的训练和推理效率。

另一种创新思路可以是**探索ConvNeXt中的大卷积核与ResNeSt的Split-Attention机制的结合**。ConvNeXt通过大卷积核扩大感受野，而ResNeSt通过分组和分裂注意力精细化特征表示 17。可以将ResNeSt块中的3x3卷积替换为ConvNeXt推荐的更大核（如5x5或7x7）的深度卷积，并调整相关的1x1卷积以适应倒置瓶颈结构。这可能使得模型既能捕获大范围的空间依赖，又能进行细粒度的多尺度特征交互。

**7.2 实验验证与论证**

对于提出的GhostECA模块，实验验证将包括以下步骤：

1. **实现GhostECA模块**：基于PyTorch，实现上述GhostECA模块的逻辑。  
2. **集成到ResNet**：将GhostECA模块替换掉基线ResNet中的BasicBlock，或者替换BasicBlock内的卷积层，形成GhostECA-ResNet。  
3. **性能对比**：  
   * 将GhostECA-ResNet与基线ResNet、ECA-ResNet、Ghost-ResNet在CIFAR-100上进行性能对比（Top-1/Top-5准确率、参数量、FLOPs）。  
   * **表7.1: GhostECA-ResNet与其它模型性能对比 (示例)**

| 模型 | Top-1 准确率 (%) | 参数量 (M) | FLOPs (M) |
| :---- | ----- | ----- | ----- |
| ResNet-20 (基线) | 65.8 | 0.27 | 40 |
| ECA-ResNet-20 | 67.5 | 0.271 | 40.05 |
| Ghost-ResNet-20 (s=2) | 66.5 | 0.15 | 22 |
| GhostECA-ResNet-20 | 67.8 | 0.16 | 23 |

4. **消融实验**：  
   * **ECA模块位置**：比较在GhostECA模块中，ECA作用于primary\_conv之后，与作用于整个GhostModule输出之后的效果。  
   * **GhostNet的ratio参数**：在GhostECA模块中，改变Ghost部分的ratio参数，观察其对整体性能和效率的影响。  
   * **与单独使用ECA或Ghost的对比**：确保GhostECA的效果优于简单地在Ghost-ResNet外部再套一个ECA层，或者在标准ResNet中仅使用ECA。

论证：  
如果实验结果（如表7.1示例）显示GhostECA-ResNet在准确率上优于Ghost-ResNet，且接近甚至超过ECA-ResNet，同时其参数量和FLOPs显著低于ECA-ResNet（并接近Ghost-ResNet），则可以论证该创新方法的有效性。它成功地结合了GhostNet的效率和ECA-Net的性能增益，实现了更好的性能-效率平衡。消融实验将进一步揭示ECA模块在Ghost结构中的最佳集成方式和GhostNet参数的适宜选择。  
这种创新性的探索不仅满足了项目的加分要求，也为设计在资源受限环境下表现优异的轻量级且强大的CNN模型提供了新的思路。

**8\. 结论与未来工作**

**8.1 结论总结**

本项目以精简版ResNet为基础，在CIFAR-100图像分类任务上，系统地实现并对比了多种先进的深度学习网络架构和注意力机制，包括ECA-Net、GhostNet以及受ConvNeXt启发的改进。重点关注了模型性能与计算效率之间的平衡。

主要实验发现总结如下：

1. **基线模型**: 建立的简化版ResNet（如ResNet-20结构）为后续比较提供了一个合理的性能基准，其训练和推理都是可行的，但性能有提升空间。  
2. **ECA-Net**: 将ECA模块集成到ResNet中（ECA-ResNet），能够在几乎不增加额外参数量和计算复杂度（FLOPs）的情况下，有效提升模型的分类准确率。这证明了其高效通道注意力机制在捕获跨通道信息方面的有效性，尤其适合对模型大小和计算开销敏感的应用 21。  
3. **GhostNet**: 通过将ResNet中的标准卷积替换为Ghost模块（Ghost-ResNet），可以显著降低模型的参数量和FLOPs，同时保持或略微提升分类准确率。GhostNet的核心思想——利用廉价操作生成冗余特征图——在提升模型效率方面表现出色，使其成为资源受限环境或嵌入式设备部署的有力候选 28。  
4. **ConvNeXt启发式改进**: 应用ConvNeXt的设计原则（如倒置瓶颈、大核深度卷积、GELU激活、LayerNorm等）对ResNet进行现代化改造，能够带来最为显著的准确率提升。然而，这种性能提升通常伴随着参数量和FLOPs的一定程度增加，需要在具体场景下权衡 17。  
5. **消融研究**: 对所选方法的关键组件进行的消融实验进一步验证了这些组件的有效性。例如，ECA-Net中自适应核大小的重要性，GhostNet中ratio参数对效率与性能的平衡作用，以及ConvNeXt各项微观设计对模型性能的贡献。  
6. **硬件适应性**: 所有方法的实现和评估都考虑了不同硬件的计算特性。FLOPs较低的模型（如Ghost-ResNet）通常具有更快的训练和推理速度。在追求高准确率的同时，必须关注模型在目标硬件上的实际运行效率。

综上所述，不同的先进技术在改进ResNet方面各有侧重：ECA-Net以极高的效率提升性能，GhostNet在大幅降低资源消耗的同时保持竞争力，而ConvNeXt的原则则能最大化挖掘模型潜力但可能带来更高的计算成本。选择哪种方法取决于对准确率和效率的具体需求。

**8.2 未来工作展望**

基于本项目的研究成果，未来可以从以下几个方面进行更深入的探索：

1. **探索更多先进技术组合**:  
   * 研究本项目未覆盖的其他先进注意力机制（如SegNeXt中的卷积注意力 73、LSKNet中的大选择核 75、HorNet中的高阶空间交互 76）或网络架构（如CoatNet 77、MLP-Mixer 47）与ResNet的结合。  
   * 进一步探索如第7节中提出的GhostECA等创新性组合模块的潜力，或者尝试将ConvNeXt的部分设计原则（如大核深度卷积）与ECA或GhostNet等轻量化技术更深度地融合。  
2. **针对不同硬件环境的优化**:  
   * **模型量化 (Quantization)**: 研究将训练好的高性能模型（如ConvNeXt-like-ResNet）进行量化（例如INT8量化），以在CPU上实现更快的推理速度，同时尽量减小精度损失。  
   * **模型剪枝 (Pruning)**: 对模型进行结构化或非结构化剪枝，去除冗余参数和连接，以进一步压缩模型大小和降低计算量。  
   * **知识蒸馏 (Knowledge Distillation)**: 使用一个更大、性能更好的教师模型来指导小型学生模型的训练，以期在保持低资源消耗的同时提升学生模型的性能。  
3. **更广泛的实验验证**:  
   * **不同规模的ResNet基线**: 将这些先进技术应用于不同深度和宽度的ResNet基线（如ResNet32、ResNet56的CIFAR版本），考察其在不同模型容量下的表现。  
   * **其他数据集**: 在其他图像分类数据集（如CIFAR-10、Tiny ImageNet，甚至更大规模的数据集如果计算资源允许）上验证这些改进方法的泛化能力。  
   * **迁移学习**: 评估这些在CIFAR-100上改进的ResNet模型作为预训练骨干网络，在其他下游视觉任务（如目标检测、语义分割的简化版本）上的迁移学习效果。  
4. **超参数优化与训练策略**:  
   * 利用更先进的超参数优化算法（如贝叶斯优化）来精细调整各模型的训练参数。  
   * 探索更复杂的学习率调度策略、数据增强方法（如AutoAugment 79、Mixup 80）对模型性能的影响，通过策略优化来加速收敛或提升最终性能。  
5. **深入分析模型行为**:  
   * 使用可视化技术（如Grad-CAM）来理解不同注意力机制和架构改进如何影响模型的决策过程和特征学习。  
   * 分析模型在不同类别上的性能差异，找出模型的"短板"类别，并尝试针对性地改进。

通过这些未来的工作，可以进一步推动高效且高性能的深度学习模型在资源受限环境下的应用和发展。

**9\. 团队成员贡献**

本项目由五人团队合作完成，各成员在项目的不同阶段和方面均做出了重要贡献。详细分工如下表所示：

**表9.1: 团队成员贡献详情**

| 成员姓名 | 主要负责任务 | 具体贡献描述 |
| :---- | :---- | :---- |
| 成员A | 项目总协调、文献调研（ConvNeXt, ResNet）、基线模型 (Simplified ResNet) 设计与实现、ConvNeXt启发式ResNet的设计与实现、实验设计（总体框架）、报告撰写（引言、背景、基线模型、ConvNeXt部分、结论） | 负责项目的整体规划与进度管理；深入研究ResNet及ConvNeXt相关文献，确定基线模型架构和ConvNeXt改进方案；独立完成基线ResNet和ConvNeXt-like ResNet的PyTorch代码实现与调试；参与设计整体实验流程和对比方案；撰写报告的关键章节，确保技术描述的准确性和完整性。 |
| 成员B | 文献调研（ECA-Net, 注意力机制）、ECA-Net模块实现与集成（ECA-ResNet）、实验执行（ECA-ResNet的训练与评估）、结果分析（ECA-Net相关）、消融实验设计（ECA-Net部分） | 重点研究ECA-Net及其他相关通道注意力机制的原理和实现；独立完成ECA模块的PyTorch代码编写，并将其成功集成到ResNet基线模型中；负责ECA-ResNet的全部训练、调参与性能评估实验；设计并执行了针对ECA模块的消融实验；分析ECA-Net的实验结果并撰写报告中对应的方法实现与结果分析章节。 |
| 成员C | 文献调研（GhostNet, 轻量化网络）、GhostNet模块实现与集成（Ghost-ResNet）、实验执行（Ghost-ResNet的训练与评估）、结果分析（GhostNet相关）、消融实验设计（GhostNet部分） | 专注于GhostNet等轻量化网络架构的研究；独立完成GhostModule和GhostBottleneck的PyTorch代码实现，并将其应用于ResNet基线；负责Ghost-ResNet的训练、超参数调整和性能测试；设计并实施了针对GhostNet关键参数（如ratio s）的消融研究；分析GhostNet的实验数据并撰写报告中相应的方法实现与结果分析章节。 |
| 成员D | 实验环境搭建与维护、数据集预处理与加载模块编写、通用训练与评估脚本编写、实验结果汇总与可视化（图表制作）、报告撰写（实验设置、实验结果汇总与对比分析）、代码库管理 | 负责搭建和维护统一的PyTorch实验环境，确保所有成员工作在一致的平台上；编写通用的CIFAR-100数据预处理、数据加载、模型训练和评估脚本，供团队成员使用；收集、整理所有模型的实验数据，并制作规范化的表格和图表用于报告和PPT展示；撰写报告中的实验设置和整体结果对比分析章节；负责项目代码的版本控制与管理。 |
| 成员E | 文献调研（CSPNet, ResNeSt, MLP-Mixer等备选方法）、（若有）创新性方法提出与初步探索、PPT演示文稿设计与制作、报告校对与格式规范、未来工作部分撰写 | 调研了多种备选的先进网络架构和注意力机制，为团队技术选型提供支持；（如果团队实现了创新方法）主导或深度参与了创新性方法的构思、文献查阅和初步实现与验证；负责将实验报告的核心内容转化为清晰、专业的PPT演示文稿；对整个实验报告进行细致的校对，确保语言流畅、格式统一、无技术性错误；撰写报告中的未来工作展望部分。 |

*(注：如果团队选择了不同的三种方法，或者实现了创新方法，请相应调整成员B、C、E的任务描述。)*

这种明确的分工与合作确保了项目能够全面、高效地完成各项研究任务，并保证了实验报告和最终成果的质量。团队成员之间的密切沟通和协作是项目成功的关键。

**10\. 参考文献**

1. Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images*. University of Toronto. 1  
2. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11976-11986). 17  
3. Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11534-11542). 21  
4. Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., & Xu, C. (2020). GhostNet: More Features From Cheap Operations. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1580-1589). 27  
5. Wang, C. Y., Liao, H. Y. M., Wu, Y. H., Chen, P. Y., Hsieh, J. W., & Yeh, I. H. (2020). CSPNet: A New Backbone that Can Enhance Learning Capability of CNN. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)* (pp. 1571-1580). 31  
6. Zhang, H., Wu, C., Zhang, Z., Zhu, Y., Lin, H., Sun, Y.,... & Smola, A. (2022). ResNeSt: Split-Attention Networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. (Original ArXiv: arXiv:2004.08955). 40  
7. Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T.,... & Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture for Vision. In *Advances in Neural Information Processing Systems (NeurIPS)*, 34\. 46  
8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)* (pp. 770-778). 3  
9. pytorch/vision GitHub Repository (for ResNet, ConvNeXt, etc. implementations). https://github.com/pytorch/vision 13  
10. huggingface/pytorch-image-models (timm) GitHub Repository. https://github.com/huggingface/pytorch-image-models 37  
11. BangguWu/ECANet GitHub Repository. https://github.com/BangguWu/ECANet 54  
12. huawei-noah/ghostnet (CV-Backbones) GitHub Repository. https://github.com/huawei-noah/ghostnet or https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet\_pytorch 29  
13. zhanghang1989/ResNeSt GitHub Repository. https://github.com/zhanghang1989/ResNeSt 45  
14. WongKinYiu/CrossStagePartialNetworks GitHub Repository. https://github.com/WongKinYiu/CrossStagePartialNetworks 105  
15. google-research/vision\_transformer GitHub Repository (for MLP-Mixer JAX impl). https://github.com/google-research/vision\_transformer 83  
16. kuangliu/pytorch-cifar GitHub Repository. https://github.com/kuangliu/pytorch-cifar 11  
17. PyTorch Tutorials (CIFAR10 classification, training loop). https://pytorch.org/tutorials/beginner/blitz/cifar10\_tutorial.html 9  
18. Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning augmentation strategies from data. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 113-123). 79  
19. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. *arXiv preprint arXiv:1710.09412*. 80  
20. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 1-48. 7  
21. PyTorch Documentation (Optimizers, Schedulers). https://pytorch.org/docs/stable/optim.html 72  
22. Krizhevsky, A., Nair, V., & Hinton, G. (2009). CIFAR-100 (Canadian Institute for Advanced Research). https://www.cs.toronto.edu/\~kriz/cifar.html 81 *(Additional papers for other selected methods like SegNeXt, LSKNet, CoatNet, HorNet would be added here if they were chosen and implemented)*

**11\. 附录 (可选)**

本附录旨在提供主报告中未能详尽展示的补充材料，包括更详细的代码实现、完整的超参数配置以及额外的实验结果图表，以便感兴趣的读者深入了解本项目的技术细节和研究过程。

**11.1 核心模块PyTorch代码片段**

* **简化版ResNet BasicBlock (针对CIFAR-100)**  
  Python  
  import torch  
  import torch.nn as nn  
  import torch.nn.functional as F

  class BasicBlock(nn.Module):  
      expansion \= 1

      def \_\_init\_\_(self, in\_planes, planes, stride=1):  
          super(BasicBlock, self).\_\_init\_\_()  
          self.conv1 \= nn.Conv2d(in\_planes, planes, kernel\_size=3, stride=stride, padding=1, bias=False)  
          self.bn1 \= nn.BatchNorm2d(planes)  
          self.conv2 \= nn.Conv2d(planes, planes, kernel\_size=3, stride=1, padding=1, bias=False)  
          self.bn2 \= nn.BatchNorm2d(planes)

          self.shortcut \= nn.Sequential()  
          if stride\!= 1 or in\_planes\!= self.expansion \* planes:  
              self.shortcut \= nn.Sequential(  
                  nn.Conv2d(in\_planes, self.expansion \* planes, kernel\_size=1, stride=stride, bias=False),  
                  nn.BatchNorm2d(self.expansion \* planes)  
              )

      def forward(self, x):  
          out \= F.relu(self.bn1(self.conv1(x)))  
          out \= self.bn2(self.conv2(out))  
          out \+= self.shortcut(x)  
          out \= F.relu(out)  
          return out

  *(可在此处补充完整的Simplified ResNet模型定义代码)*  
* ECA模块 (eca\_layer)  
  (参见报告第4.1.1节中的代码示例)  
* Ghost模块 (GhostModule)  
  (参见报告第4.2.1节中的代码示例)  
* ConvNeXt风格块 (ConvNeXtLikeBlock \- 概念性)  
  由于ConvNeXt的集成是对ResNet的深度改造，此处不提供单一模块代码，其核心思想已在4.3.1节阐述。具体实现可参考timm库中ConvNeXtBlock的结构，并将其适配到ResNet的块替换中。关键组件包括nn.Conv2d (用于深度卷积 groups=in\_channels 和点卷积 kernel\_size=1)，nn.LayerNorm，nn.GELU。

**11.2 完整超参数表**

**表11.1: 各模型训练超参数详情**

| 超参数 | 基线ResNet | ECA-ResNet | Ghost-ResNet | ConvNeXt-like-ResNet | (创新方法-如有) |
| :---- | ----- | ----- | ----- | ----- | ----- |
| 优化器 | SGD | SGD | SGD | AdamW | SGD |
| 初始学习率 | 0.1 | 0.1 | 0.1 | 0.001 | 0.1 |
| 学习率调度 | CosineAnneal | CosineAnneal | CosineAnneal | CosineAnneal | CosineAnneal |
| Momentum (SGD) | 0.9 | 0.9 | 0.9 | N/A | 0.9 |
| Weight Decay | 5e-4 | 5e-4 | 5e-4 | 0.05 | 5e-4 |
| Batch Size | 128 | 128 | 128 | 128 | 128 |
| Epochs | 150 | 150 | 150 | 150 | 150 |
| ECA: k\_size | N/A | 自适应 | N/A | N/A | 自适应(如有) |
| GhostNet: ratio(s) | N/A | N/A | 2 | N/A | 2(如有) |

*(注: 此表为示例，具体数值应根据实际实验填写。ConvNeXt-like-ResNet的AdamW权重衰减通常设置较高，如0.05。)*

**11.3 额外图表**

* **训练/验证损失曲线**: 为每个主要模型（基线、ECA-ResNet、Ghost-ResNet、ConvNeXt-like-ResNet）提供训练损失和验证损失随训练轮次变化的曲线图。这有助于分析模型的收敛情况和是否存在过拟合。  
  (示例图位置：可在此处嵌入损失曲线图的图片或描述)  
  图11.1: 基线ResNet-20在CIFAR-100上的训练与验证损失曲线。  
  图11.2: ECA-ResNet-20在CIFAR-100上的训练与验证损失曲线。  
  ...  
* **训练/验证准确率曲线**: 为每个主要模型提供训练准确率和验证准确率随训练轮次变化的曲线图。  
  (示例图位置：可在此处嵌入准确率曲线图的图片或描述)  
  图11.3: 基线ResNet-20在CIFAR-100上的训练与验证准确率曲线。  
  ...  
* **部分类别混淆矩阵 (可选)**: 对于表现差异较大的模型，可以展示其在部分易混淆类别上的混淆矩阵，以更细致地分析模型的分类行为。  
  *(示例图位置：可在此处嵌入混淆矩阵图的图片或描述)*  
* **消融实验详细结果图表 (可选)**: 如果主报告中的消融实验表格过于简化，可以在附录中提供更详细的图表，例如不同ECA核大小k对各层性能影响的柱状图等。

这些补充材料为对本项目技术细节和实验过程感兴趣的读者提供了更深入的信息。

#### **Works cited**

1. ultralytics/docs/en/datasets/classify/cifar100.md at main \- GitHub, accessed May 28, 2025, [https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/classify/cifar100.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/classify/cifar100.md)  
2. CIFAR 100 Dataset \- Machine Learning Datasets \- Activeloop, accessed May 28, 2025, [https://datasets.activeloop.ai/docs/ml/datasets/cifar-100-dataset/](https://datasets.activeloop.ai/docs/ml/datasets/cifar-100-dataset/)  
3. ResNet18 from Scratch Using PyTorch \- GeeksforGeeks, accessed May 28, 2025, [https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/](https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/)  
4. Deep Learning Architectures Explained: ResNet, InceptionV3, SqueezeNet | DigitalOcean, accessed May 28, 2025, [https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet](https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet)  
5. CIFAR-100: Benchmark Dataset for Image Classification \- ModelNova, accessed May 28, 2025, [https://modelnova.ai/datasets/details/cifar-100](https://modelnova.ai/datasets/details/cifar-100)  
6. Data Preprocessing in PyTorch | GeeksforGeeks, accessed May 28, 2025, [https://www.geeksforgeeks.org/data-preprocessing-in-pytorch/](https://www.geeksforgeeks.org/data-preprocessing-in-pytorch/)  
7. Edge AI \- W2 \- CIFAR 100 \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/code/jopyth/edge-ai-w2-cifar-100](https://www.kaggle.com/code/jopyth/edge-ai-w2-cifar-100)  
8. CIFAR-100 Resnet PyTorch 75.17% Accuracy \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy](https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy)  
9. Training a Classifier — PyTorch Tutorials 2.7.0+cu126 documentation, accessed May 28, 2025, [https://docs.pytorch.org/tutorials/beginner/blitz/cifar10\_tutorial.html](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)  
10. Soft Augmentation for Image Classification \- arXiv, accessed May 28, 2025, [https://arxiv.org/html/2211.04625v2](https://arxiv.org/html/2211.04625v2)  
11. ResNet Implementation for CIFAR100 in Pytorch \- GitHub, accessed May 28, 2025, [https://github.com/fcakyon/cifar100-resnet](https://github.com/fcakyon/cifar100-resnet)  
12. Top-1 Acc (%)↑ of image classifica- tion on ImageNet-1k based on ResNet variants using PyTorch-style training recipe. \- ResearchGate, accessed May 28, 2025, [https://www.researchgate.net/figure/Top-1-Acc-of-image-classifica-tion-on-ImageNet-1k-based-on-ResNet-variants-using\_tbl1\_359390424](https://www.researchgate.net/figure/Top-1-Acc-of-image-classifica-tion-on-ImageNet-1k-based-on-ResNet-variants-using_tbl1_359390424)  
13. Source code for torchvision.models.resnet \- PyTorch documentation, accessed May 28, 2025, [https://docs.pytorch.org/vision/0.8/\_modules/torchvision/models/resnet.html](https://docs.pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html)  
14. ResNet-PyTorch/examples/cifar/README.md · main · kelm / pytorch-image-models \- GitLab, accessed May 28, 2025, [https://git.informatik.uni-hamburg.de/kelm/pytorch-image-models/-/blob/main/ResNet-PyTorch/examples/cifar/README.md](https://git.informatik.uni-hamburg.de/kelm/pytorch-image-models/-/blob/main/ResNet-PyTorch/examples/cifar/README.md)  
15. akamaster/pytorch\_resnet\_cifar10: Proper implementation of ResNet-s for CIFAR10/100 in pytorch that matches description of the original paper. \- GitHub, accessed May 28, 2025, [https://github.com/akamaster/pytorch\_resnet\_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)  
16. pytorch-cifar100/models/resnet.py at master \- GitHub, accessed May 28, 2025, [https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)  
17. Explaining ConvNeXt: A Cutting-Edge ConvNet for the 2020s \- Toolify.ai, accessed May 28, 2025, [https://www.toolify.ai/ai-news/explaining-convnext-a-cuttingedge-convnet-for-the-2020s-552093](https://www.toolify.ai/ai-news/explaining-convnext-a-cuttingedge-convnet-for-the-2020s-552093)  
18. \[2201.03545\] A ConvNet for the 2020s \- arXiv, accessed May 28, 2025, [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)  
19. openaccess.thecvf.com, accessed May 28, 2025, [https://openaccess.thecvf.com/content/CVPR2022/papers/Liu\_A\_ConvNet\_for\_the\_2020s\_CVPR\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)  
20. Achieving 3D Attention via Triplet Squeeze and Excitation Block \- arXiv, accessed May 28, 2025, [https://arxiv.org/html/2505.05943v1](https://arxiv.org/html/2505.05943v1)  
21. openaccess.thecvf.com, accessed May 28, 2025, [https://openaccess.thecvf.com/content\_CVPR\_2020/papers/Wang\_ECA-Net\_Efficient\_Channel\_Attention\_for\_Deep\_Convolutional\_Neural\_Networks\_CVPR\_2020\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf)  
22. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks \- CVPR 2020 Open Access Repository \- The Computer Vision Foundation, accessed May 28, 2025, [https://openaccess.thecvf.com/content\_CVPR\_2020/html/Wang\_ECA-Net\_Efficient\_Channel\_Attention\_for\_Deep\_Convolutional\_Neural\_Networks\_CVPR\_2020\_paper.html](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)  
23. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks \- Wandb, accessed May 28, 2025, [https://wandb.ai/diganta/ECANet-sweep/reports/ECA-Net-Efficient-Channel-Attention-for-Deep-Convolutional-Neural-Networks--VmlldzozODU0NTM](https://wandb.ai/diganta/ECANet-sweep/reports/ECA-Net-Efficient-Channel-Attention-for-Deep-Convolutional-Neural-Networks--VmlldzozODU0NTM)  
24. (PDF) ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, accessed May 28, 2025, [https://www.researchgate.net/publication/336361781\_ECA-Net\_Efficient\_Channel\_Attention\_for\_Deep\_Convolutional\_Neural\_Networks](https://www.researchgate.net/publication/336361781_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks)  
25. Efficient Channel Attention \- SERP AI, accessed May 28, 2025, [https://serp.ai/efficient-channel-attention/](https://serp.ai/efficient-channel-attention/)  
26. Performance-Efficiency Comparisons of Channel Attention Modules for ResNets \- Pure, accessed May 28, 2025, [https://pure.tue.nl/ws/portalfiles/portal/309586057/s11063-023-11161-z.pdf](https://pure.tue.nl/ws/portalfiles/portal/309586057/s11063-023-11161-z.pdf)  
27. GhostNet: More Features From Cheap Operations | Request PDF \- ResearchGate, accessed May 28, 2025, [https://www.researchgate.net/publication/343456682\_GhostNet\_More\_Features\_From\_Cheap\_Operations](https://www.researchgate.net/publication/343456682_GhostNet_More_Features_From_Cheap_Operations)  
28. openaccess.thecvf.com, accessed May 28, 2025, [https://openaccess.thecvf.com/content\_CVPR\_2020/papers/Han\_GhostNet\_More\_Features\_From\_Cheap\_Operations\_CVPR\_2020\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf)  
29. GhostNet \- PyTorch, accessed May 28, 2025, [https://pytorch.org/hub/pytorch\_vision\_ghostnet/](https://pytorch.org/hub/pytorch_vision_ghostnet/)  
30. An enhanced GhostNet model for emotion recognition: leveraging efficient feature extraction and attention mechanisms \- PMC, accessed May 28, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12016667/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12016667/)  
31. CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification \- arXiv, accessed May 28, 2025, [https://arxiv.org/html/2411.11879v1](https://arxiv.org/html/2411.11879v1)  
32. CSPNet: A New Backbone that can Enhance Learning Capability of CNN \- ResearchGate, accessed May 28, 2025, [https://www.researchgate.net/publication/337590017\_CSPNet\_A\_New\_Backbone\_that\_can\_Enhance\_Learning\_Capability\_of\_CNN](https://www.researchgate.net/publication/337590017_CSPNet_A_New_Backbone_that_can_Enhance_Learning_Capability_of_CNN)  
33. CSPNet: A New Backbone That Can Enhance Learning Capability of CNN \- CVF Open Access, accessed May 28, 2025, [https://openaccess.thecvf.com/content\_CVPRW\_2020/papers/w28/Wang\_CSPNet\_A\_New\_Backbone\_That\_Can\_Enhance\_Learning\_Capability\_of\_CVPRW\_2020\_paper.pdf](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf)  
34. accessed January 1, 1970, [https://openaccess.thecvf.com/content/CVPRW/2020/papers/w28/Wang\_CSPNet\_A\_New\_Backbone\_That\_Can\_Enhance\_Learning\_Capability\_of\_CVPRW\_2020\_paper.pdf](https://openaccess.thecvf.com/content/CVPRW/2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf)  
35. CSPNet: A New Backbone that can Enhance Learning Capability of ..., accessed May 28, 2025, [https://paperswithcode.com/paper/cspnet-a-new-backbone-that-can-enhance](https://paperswithcode.com/paper/cspnet-a-new-backbone-that-can-enhance)  
36. A Cross-Stage Partial Network and a Cross-Attention-Based Transformer for an Electrocardiogram-Based Cardiovascular Disease Decision System \- MDPI, accessed May 28, 2025, [https://www.mdpi.com/2306-5354/11/6/549](https://www.mdpi.com/2306-5354/11/6/549)  
37. pytorch-image-models/timm/models/cspnet.py at main · huggingface ..., accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cspnet.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cspnet.py)  
38. Guide on YOLOv11 Model Building from Scratch using PyTorch \- Analytics Vidhya, accessed May 28, 2025, [https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/](https://www.analyticsvidhya.com/blog/2025/01/yolov11-model-building/)  
39. 2019 CSPNet paper summary \- YouTube, accessed May 28, 2025, [https://m.youtube.com/watch?v=a0sxeZALxzY\&pp=ygUMI3N1bW1hcnkyMDE5](https://m.youtube.com/watch?v=a0sxeZALxzY&pp=ygUMI3N1bW1hcnkyMDE5)  
40. Amazon Introduces ResNeSt: Strong, Split-Attention Networks \- Synced Review, accessed May 28, 2025, [https://syncedreview.com/2020/04/24/amazon-introduces-resnest-strong-split-attention-networks/](https://syncedreview.com/2020/04/24/amazon-introduces-resnest-strong-split-attention-networks/)  
41. (PDF) ResNeSt: Split-Attention Networks \- ResearchGate, accessed May 28, 2025, [https://www.researchgate.net/publication/340805846\_ResNeSt\_Split-Attention\_Networks](https://www.researchgate.net/publication/340805846_ResNeSt_Split-Attention_Networks)  
42. ResNeSt: Split-Attention Networks \- Hang Zhang, accessed May 28, 2025, [https://hangzhang.org/files/resnest.pdf](https://hangzhang.org/files/resnest.pdf)  
43. openaccess.thecvf.com, accessed May 28, 2025, [https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang\_ResNeSt\_Split-Attention\_Networks\_CVPRW\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf)  
44. resnest.mdx \- huggingface/pytorch-image-models \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/resnest.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/resnest.mdx)  
45. ResNeSt \- PyTorch, accessed May 28, 2025, [https://pytorch.org/hub/pytorch\_vision\_resnest/](https://pytorch.org/hub/pytorch_vision_resnest/)  
46. Pay Attention to MLPs \- ResearchGate, accessed May 28, 2025, [https://www.researchgate.net/publication/351656442\_Pay\_Attention\_to\_MLPs](https://www.researchgate.net/publication/351656442_Pay_Attention_to_MLPs)  
47. MLP-Mixer: An all-MLP Architecture for Vision \- NIPS, accessed May 28, 2025, [https://proceedings.nips.cc/paper\_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf](https://proceedings.nips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf)  
48. MLP-Mixer: An all-MLP Architecture for Vision, accessed May 28, 2025, [https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html)  
49. \[21.05\] MLP-Mixer \- DOCSAID, accessed May 28, 2025, [https://docsaid.org/en/papers/vision-transformers/mlp-mixer/](https://docsaid.org/en/papers/vision-transformers/mlp-mixer/)  
50. accessed January 1, 1970, [https://proceedings.neurips.cc/paper/2021/file/4ccb653b2c3537e5d1e917d413c686ff-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/4ccb653b2c3537e5d1e917d413c686ff-Paper.pdf)  
51. proceedings.neurips.cc, accessed May 28, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf)  
52. A simple implementation of MLP Mixer in Pytorch \- GitHub, accessed May 28, 2025, [https://github.com/rrmina/MLP-Mixer-pytorch](https://github.com/rrmina/MLP-Mixer-pytorch)  
53. pytorch-cifar/models/resnet.py at master · kuangliu/pytorch-cifar ..., accessed May 28, 2025, [https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)  
54. BangguWu/ECANet: Code for ECA-Net: Efficient Channel ... \- GitHub, accessed May 28, 2025, [https://github.com/BangguWu/ECANet](https://github.com/BangguWu/ECANet)  
55. ECA-ResNet \- Pytorch Image Models, accessed May 28, 2025, [https://pprp.github.io/timm/models/ecaresnet/](https://pprp.github.io/timm/models/ecaresnet/)  
56. ECANet/models/eca\_module.py at master · BangguWu/ECANet ..., accessed May 28, 2025, [https://github.com/BangguWu/ECANet/blob/master/models/eca\_module.py](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py)  
57. pytorch-image-models/timm/models/ghostnet.py at main \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/master/timm/models/ghostnet.py](https://github.com/huggingface/pytorch-image-models/blob/master/timm/models/ghostnet.py)  
58. huawei-noah/Efficient-AI-Backbones: Efficient AI ... \- GitHub, accessed May 28, 2025, [https://github.com/huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet)  
59. Efficient-AI-Backbones/ghostnet\_pytorch/ghostnet.py at master ..., accessed May 28, 2025, [https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet\_pytorch/ghostnet.py](https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet_pytorch/ghostnet.py)  
60. pytorch-image-models/timm/models/ghostnet.py at main \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/ghostnet.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/ghostnet.py)  
61. FASTER CONVOLUTIONAL NEURAL NETWORKS TRAINING by Shanshan Jiang \- Niner Commons, accessed May 28, 2025, [https://ninercommons.charlotte.edu/record/2007/files/Jiang\_uncc\_0694D\_12888.pdf](https://ninercommons.charlotte.edu/record/2007/files/Jiang_uncc_0694D_12888.pdf)  
62. ConvNeXt \- MMPretrain 1.2.0 documentation \- Read the Docs, accessed May 28, 2025, [https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ConvNeXt.html](https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ConvNeXt.html)  
63. FrancescoSaverioZuppichini/ConvNext: Implementing ConvNext in PyTorch \- GitHub, accessed May 28, 2025, [https://github.com/FrancescoSaverioZuppichini/ConvNext](https://github.com/FrancescoSaverioZuppichini/ConvNext)  
64. Source code for torchvision.models.convnext \- PyTorch documentation, accessed May 28, 2025, [https://docs.pytorch.org/vision/main/\_modules/torchvision/models/convnext.html](https://docs.pytorch.org/vision/main/_modules/torchvision/models/convnext.html)  
65. Source code for torchvision.models.convnext \- PyTorch documentation, accessed May 28, 2025, [https://docs.pytorch.org/vision/0.12/\_modules/torchvision/models/convnext.html](https://docs.pytorch.org/vision/0.12/_modules/torchvision/models/convnext.html)  
66. facebookresearch/ConvNeXt: Code release for ConvNeXt ... \- GitHub, accessed May 28, 2025, [https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)  
67. ConvNeXt/models/convnext.py at main · facebookresearch ... \- GitHub, accessed May 28, 2025, [https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)  
68. accessed January 1, 1970, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py)  
69. Exploring the Impact of Architectural Variations in ResNet on CIFAR-100 Performance: An Investigation on Fully Connected Layers, \- Dean & Francis, accessed May 28, 2025, [https://www.deanfrancispress.com/index.php/te/article/download/1708/TE003232.pdf/5548](https://www.deanfrancispress.com/index.php/te/article/download/1708/TE003232.pdf/5548)  
70. Transforming and augmenting images — Torchvision main documentation, accessed May 28, 2025, [https://docs.pytorch.org/vision/main/transforms.html](https://docs.pytorch.org/vision/main/transforms.html)  
71. Appendix: On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them, accessed May 28, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/040d3b6af368bf71f952c18da5713b48-Supplemental-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/040d3b6af368bf71f952c18da5713b48-Supplemental-Conference.pdf)  
72. torch.optim — PyTorch 2.7 documentation, accessed May 28, 2025, [https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html)  
73. SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation \- Tsinghua Graphics and Geometric Computing Group, accessed May 28, 2025, [https://cg.cs.tsinghua.edu.cn/papers/NeurIPS-2022-SegNeXt.pdf](https://cg.cs.tsinghua.edu.cn/papers/NeurIPS-2022-SegNeXt.pdf)  
74. SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation, accessed May 28, 2025, [https://mftp.mmcheng.net/Papers/22NeurIPS-SegNeXt.pdf](https://mftp.mmcheng.net/Papers/22NeurIPS-SegNeXt.pdf)  
75. LSKNet: A Foundation Lightweight Backbone for Remote Sensing \- arXiv, accessed May 28, 2025, [https://arxiv.org/html/2403.11735v5](https://arxiv.org/html/2403.11735v5)  
76. HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions \- OpenReview, accessed May 28, 2025, [https://openreview.net/pdf?id=tro0\_OqIVde](https://openreview.net/pdf?id=tro0_OqIVde)  
77. CoAtNet: Marrying Convolution and Attention for All Data Sizes \- OpenReview, accessed May 28, 2025, [https://openreview.net/pdf?id=dUk5Foj5CLf](https://openreview.net/pdf?id=dUk5Foj5CLf)  
78. CoAtNet: Marrying Convolution and Attention for All Data Sizes \- arXiv, accessed May 28, 2025, [https://arxiv.org/pdf/2106.04803](https://arxiv.org/pdf/2106.04803)  
79. Comparing Different Automatic Image Augmentation Methods in PyTorch, accessed May 28, 2025, [https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html](https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html)  
80. Mixup CIFAR-10/100 Benchmarks \- OpenMixup documentation, accessed May 28, 2025, [https://openmixup.readthedocs.io/en/latest/mixup\_benchmarks/Mixup\_cifar.html](https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_cifar.html)  
81. CIFAR-10 and CIFAR-100 datasets, accessed May 28, 2025, [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)  
82. accessed January 1, 1970, [https://openaccess.thecvf.com/content/CVPR2020/papers/Wang\_ECA-Net\_Efficient\_Channel\_Attention\_for\_Deep\_Convolutional\_Neural\_Networks\_CVPR\_2020\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf)  
83. MLP-Mixer: An all-MLP Architecture for Vision | Papers With Code, accessed May 28, 2025, [https://paperswithcode.com/paper/mlp-mixer-an-all-mlp-architecture-for-vision](https://paperswithcode.com/paper/mlp-mixer-an-all-mlp-architecture-for-vision)  
84. CSP-ResNet \- Pytorch Image Models, accessed May 28, 2025, [https://pprp.github.io/timm/models/csp-resnet/](https://pprp.github.io/timm/models/csp-resnet/)  
85. Starter: pytorch image models 5341d578-a \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/code/kerneler/starter-pytorch-image-models-5341d578-a/data](https://www.kaggle.com/code/kerneler/starter-pytorch-image-models-5341d578-a/data)  
86. PyTorch Image Models \- timm · PyPI, accessed May 28, 2025, [https://pypi.org/project/timm/0.8.2.dev0/](https://pypi.org/project/timm/0.8.2.dev0/)  
87. csp-resnet.mdx \- huggingface/pytorch-image-models \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/csp-resnet.mdx](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/csp-resnet.mdx)  
88. CIFAR10 convnext \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/code/stpeteishii/cifar10-convnext](https://www.kaggle.com/code/stpeteishii/cifar10-convnext)  
89. CIFAR-100 ConvNeXT \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/code/mtamer1418/cifar-100-convnext](https://www.kaggle.com/code/mtamer1418/cifar-100-convnext)  
90. composer.models.timm.model \- Databricks Mosaic AI Training, accessed May 28, 2025, [https://docs.mosaicml.com/projects/composer/en/v0.13.5/\_modules/composer/models/timm/model.html](https://docs.mosaicml.com/projects/composer/en/v0.13.5/_modules/composer/models/timm/model.html)  
91. pytorch\_image\_models \- Kaggle, accessed May 28, 2025, [https://www.kaggle.com/datasets/kozistr/pytorch-image-models](https://www.kaggle.com/datasets/kozistr/pytorch-image-models)  
92. MadryLab/timm-bench: PyTorch image models, scripts, pretrained weights \-- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more \- GitHub, accessed May 28, 2025, [https://github.com/MadryLab/timm-bench](https://github.com/MadryLab/timm-bench)  
93. convnext\_tiny — Torchvision main documentation, accessed May 28, 2025, [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext\_tiny.html](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html)  
94. timm/convnext\_tiny.in12k \- Hugging Face, accessed May 28, 2025, [https://huggingface.co/timm/convnext\_tiny.in12k](https://huggingface.co/timm/convnext_tiny.in12k)  
95. timm/ghostnet\_100.in1k \- Hugging Face, accessed May 28, 2025, [https://huggingface.co/timm/ghostnet\_100.in1k](https://huggingface.co/timm/ghostnet_100.in1k)  
96. timm/resnest50d.in1k \- Hugging Face, accessed May 28, 2025, [https://huggingface.co/timm/resnest50d.in1k](https://huggingface.co/timm/resnest50d.in1k)  
97. pytorch-image-models/timm/models/resnest.py at main \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnest.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnest.py)  
98. accessed January 1, 1970, [https://github.com/huggingface/pytorch-image-models/tree/main/timm/models](https://github.com/huggingface/pytorch-image-models/tree/main/timm/models)  
99. github.com, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py)  
100. accessed January 1, 1970, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eca.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eca.py)  
101. pytorch-image-models/timm/models/mlp\_mixer.py at main \- GitHub, accessed May 28, 2025, [https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp\_mixer.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py)  
102. Unofficial Implementation of ECANets (CVPR 2020\) for the Reproducibility Challenge 2020\. \- GitHub, accessed May 28, 2025, [https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET](https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET)  
103. zhanghang1989/ResNeSt: ResNeSt: Split-Attention Networks \- GitHub, accessed May 28, 2025, [https://github.com/zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)  
104. ResNeSt/resnest/torch/models/resnest.py at master ... \- GitHub, accessed May 28, 2025, [https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py](https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/resnest.py)  
105. WongKinYiu/CrossStagePartialNetworks: Cross Stage ... \- GitHub, accessed May 28, 2025, [https://github.com/WongKinYiu/CrossStagePartialNetworks](https://github.com/WongKinYiu/CrossStagePartialNetworks)  
106. accessed January 1, 1970, [https://github.com/google-research/vision\_transformer/tree/main/vit\_jax/models\_mixer.py](https://github.com/google-research/vision_transformer/tree/main/vit_jax/models_mixer.py)  
107. Top Highly Used Repositories \- OpenMeter, accessed May 28, 2025, [https://openmeter.benchcouncil.org/search?q=pytorch](https://openmeter.benchcouncil.org/search?q=pytorch)  
108. kuangliu/pytorch-cifar: 95.47% on CIFAR10 with PyTorch \- GitHub, accessed May 28, 2025, [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)  
109. Training a Classifier — PyTorch Tutorials 2.7.0+cu126 documentation, accessed May 28, 2025, [https://pytorch.org/tutorials/beginner/blitz/cifar10\_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)  
110. CIFAR-10 \- Wikipedia, accessed May 28, 2025, [https://en.wikipedia.org/wiki/CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)