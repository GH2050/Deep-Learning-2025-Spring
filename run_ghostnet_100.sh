#!/bin/bash

echo "开始训练 GhostNet-100 模型"
echo "========================================="

source /opt/venvs/base/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --config_file default_config.yaml \
    --main_process_port 29501 \
    train_ghostnet_100.py

echo "训练完成!"
echo "请查看 logs/ghostnet_100/ 目录下的结果文件" 