#!/bin/bash

source /root/data-tmp/miniconda3/etc/profile.d/conda.sh
conda activate llm

apt-get update && apt-get install -y fonts-wqy-zenhei
rm -rf /root/.cache/matplotlib # 清除matplotlib字体缓存

# 自动检测GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

# 生成GPU列表 (0,1,2,...)
GPU_LIST=$(seq -s, 0 $((GPU_COUNT-1)))
export CUDA_VISIBLE_DEVICES=$GPU_LIST

MODEL_NAME=${1:-"resnet_56"}
EPOCHS=${2:-300}
# BATCH_SIZE is now the 3rd argument, can be omitted
# LR is now the 4th argument, can be omitted

echo "开始训练 ${MODEL_NAME}..."
echo "使用GPU: $GPU_LIST"

CMD_ARGS="--model_name $MODEL_NAME --epochs $EPOCHS"

# Handle Batch Size (3rd argument)
if [ -n "$3" ]; then
    BATCH_SIZE=$3
    echo "批次大小: $BATCH_SIZE (来自命令行)"
    CMD_ARGS="$CMD_ARGS --batch_size $BATCH_SIZE"
else
    echo "批次大小: 使用模型默认值 (来自src/utils.py)"
fi

# Handle Learning Rate (4th argument)
if [ -n "$4" ]; then
    LR=$4
    echo "学习率: $LR (来自命令行)"
    CMD_ARGS="$CMD_ARGS --lr $LR"
else
    echo "学习率: 使用模型默认值 (来自src/utils.py)"
fi

echo "轮数: $EPOCHS"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node=$GPU_COUNT \
    --master_port=29502 \
    src/train.py \
    $CMD_ARGS

echo "训练完成！" 