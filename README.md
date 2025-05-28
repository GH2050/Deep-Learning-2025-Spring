# DL-2025 项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目目标

使用 PyTorch 实现 CIFAR-100 分类任务，基于精简版 ResNet 作为基础网络，探索并对比多种先进的深度学习网络架构或注意力机制。

**技术栈要求**：
- **PyTorch**：主要深度学习框架
- **datasets**：数据集加载和处理
- **accelerate**：训练加速和优化
- **huggingface_hub**：模型和数据资源管理
- **transformers**：预训练模型
  
## hugginface 镜像配置

```python
os.environ['HF_ENDPOINT'] = 'https://hf-api.gitee.com'
os.environ['HF_HOME'] = 'your/path'
```

## quick start

```python
pip install -r requirements.txt
```

## 核心要求

1. **基础模型**: 采用精简版 ResNet。
2. **技术选型**: 从提供的 10 种方法中至少选择并实现 3 种，进行组合或改进。
3. **团队合作**: 以 5 人为一组进行。

## 实现过程

- 详细介绍基础方法 (ResNet) 及其改进点。
- 实现所选方法，并集成到基础网络中。
- 提供实验对比结果和消融性实验 (Ablation Studies)。

## 成果展示

- 通过 PPT 展示，时长 8-10 分钟。
- 内容包括基础方法、改进方案、实验结果和未来展望。

## 实验报告

- 提交详细报告，明确团队成员的具体贡献。

## 加分项

- 提出创新性方法并通过实验验证。

## 参考论文

详见 `requirement.md` 中的论文链接。