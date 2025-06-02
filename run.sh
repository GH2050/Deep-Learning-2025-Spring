#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_NAME=${1:-"resnet_56"}
EPOCHS=${2:-300}
BATCH_SIZE=${3:-128}
LR=${4:-0.1}

echo "开始训练 ${MODEL_NAME}..."
echo "轮数: $EPOCHS, 批次大小: $BATCH_SIZE, 学习率: $LR"

torchrun --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo "训练完成！" 