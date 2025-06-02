# DL-2025

## 📝 项目核心要求

使用 PyTorch 实现 CIFAR-100 分类。
在精简版的 ResNet 作为基础网络之上，探索、实现并对比多种先进的深度学习网络架构或注意力机制。


**技术栈要求**：

- **PyTorch**：主要深度学习框架，用于模型构建和训练
- **datasets**：用于数据集加载、预处理和管理
- **accelerate**：用于训练加速、分布式训练和性能优化    
- **huggingface_hub**：用于模型版本管理和资源共享
- **transformers**：仅使用基础构件（优化器、学习率调度器、工具函数等），不依赖预训练模型

---

## 🛠️ 实现过程要求

1.  **基础模型**: 采用精简版 ResNet。
2.  **技术选型与参考论文**:
    实现以下列表中全部10种不同的方法（可以一种或多种进行组合）。以下是这些方法的具体名称和对应的参考论文：

        1.  **ConvNeXt**: A ConvNet for the 2020s
            * 论文: `https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf`
        2.  **SegNeXt**: Rethinking convolutional attention design for semantic segmentation
            * 论文: `https://proceedings.neurips.cc/paper_files/paper/2022/file/005f5f40f4416f1ecfc3080e60b38f1a-Paper-Conference.pdf`
        3.  **LSKNet**: Large selective kernel network for remote sensing object detection
            * 论文: `https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf`
        4.  **CoatNet**: Marrying convolution and attention for all data sizes
            * 论文: `https://proceedings.neurips.cc/paper/2021/file/20598b92d6e43456acb242c1f9-Paper.pdf`
        5.  **ECA-Net**: Efficient channel attention for deep convolutional neural networks
            * 论文: `https://openaccess.thecvf.com/content/CVPR2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf`
        6.  **CSPNet**: A new backbone that can enhance learning capability of CNN
            * 论文: `https://openaccess.thecvf.com/content/CVPRW/2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf`
        7.  **GhostNet**: More features from cheap operations
            * 论文: `https://openaccess.thecvf.com/content/CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf`
        8.  **HorNet**: Efficient high-order spatial interactions with recursive gated convolutions
            * 论文: `https://proceedings.neurips.cc/paper_files/paper/2022/file/436d42b3dd81214023ae43b96b146-Paper-Conference.pdf`
        9.  **ResNeSt**: Split-attention networks
            * 论文: `https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf`
        10. **Pay Attention to MLPs** (MLP-Mixer)
            * 论文: `https://proceedings.neurips.cc/paper/2021/file/4ccb653b2c3537e5d1e917d413c686ff-Paper.pdf`

你需要考虑推荐的三种或多种选择。

3.  **团队合作**: 以 **5 人**为一组进行。

---

## 📊 实验与对比要求

* 介绍所使用的基础方法 (ResNet)。
* 详细说明如何对基础方法进行改进（即如何实现和集成所选的 3+ 种方法）。
* 提供详细的**实验对比结果**。
* 进行必要的**消融性实验** (Ablation Studies) 来验证各个模块或改进点的有效性。

---

## 🎤 成果展示要求

* **形式**: 采用 PPT 方式进行展示。
* **时长**: 每组展示时间**约为 8-10 分钟**。
* **内容**:
    * 基础方法介绍。
    * 改进方案详述。
    * 实验对比与消融结果分析。
    * 对未来工作的设想和展望。

---

## 📖 实验报告要求

* 提交一份详细的实验报告。
* 报告中必须**详细阐述团队中每个成员的具体贡献**（例如：文档撰写、实验设计与执行、Idea 提出等）。

---

## ✨ 加分项 (最多 5 分)

* 在优秀完成基础要求之上，**提出创新性的方法**。
* 通过**实验**对提出的创新方法进行**验证和论证**。