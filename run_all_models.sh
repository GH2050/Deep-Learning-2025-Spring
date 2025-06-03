#!/bin/bash

# 模型列表
models=(
    # "resnet_20"
    # "resnet_32"
    # "resnet_56"
    # "eca_resnet_20"
    # "eca_resnet_32"
    # "ghost_resnet_20"
    # "ghost_resnet_32"
    # "ghostnet_100"
    "convnext_tiny"
    # "improved_resnet20_convnext"
    "segnext_mscan_tiny"
    "mlp_mixer_tiny"
    "mlp_mixer_b16"
    "cspresnet50"
    "resnest50d"
    "coatnet_0"
    "coatnet_0_custom_enhanced"
    # "coatnet_cifar_opt"
    # "coatnet_cifar_opt_large_stem"
    # "ecanet20_fixed_k3"
    # "ecanet20_fixed_k5"
    # "ecanet20_fixed_k7"
    # "ecanet20_fixed_k9"
    # "ecanet20_adaptive"
    # "resnet20_no_eca"
)

# 依次执行模型训练
for model_name in "${models[@]}"
do
    echo "----------------------------------------------------"
    echo "开始训练模型: $model_name"
    echo "----------------------------------------------------"
    bash run.sh "$model_name"
    
    if [ $? -ne 0 ]; then
        echo "----------------------------------------------------"
        echo "模型 $model_name 训练失败，跳过..."
        echo "----------------------------------------------------"
    else
        echo "----------------------------------------------------"
        echo "模型 $model_name 训练完成。"
        echo "----------------------------------------------------"
    fi
done

echo "所有模型训练尝试完毕。"

echo "----------------------------------------------------"
echo "开始运行所有消融实验..."
echo "----------------------------------------------------"

# 确保使用正确的python解释器和激活的环境
# 例如: /opt/venvs/base/bin/python src/ablation_experiments.py
# 或者，如果环境已激活，可以直接调用 python
python3 src/ablation_experiments.py

if [ $? -ne 0 ]; then
    echo "----------------------------------------------------"
    echo "消融实验运行失败。"
    echo "----------------------------------------------------"
else
    echo "----------------------------------------------------"
    echo "所有消融实验运行完成。"
    echo "----------------------------------------------------"
fi

echo "所有实验已尝试运行。" 