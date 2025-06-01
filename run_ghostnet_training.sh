#!/bin/bash

echo "激活llm环境并开始训练GhostNet-100模型..."

source ~/miniconda3/etc/profile.d/conda.sh

conda activate llm

if [ $? -ne 0 ]; then
    echo "错误: 无法激活llm环境"
    exit 1
fi

echo "当前Python版本:"
python --version

echo "当前环境:"
conda info --envs | grep \*

echo "开始运行训练脚本..."
python train_ghostnet_100.py

echo "训练完成!" 