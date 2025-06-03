#!/usr/bin/env python3 # 保持shebang，虽然-m运行时可能不直接用
import torch
# import torch.nn as nn # No longer directly used here
# import torch.optim as optim # No longer directly used here
# from torch.utils.data import DataLoader # No longer directly used here
# import torchvision # No longer directly used here
# import torchvision.transforms as transforms # No longer directly used here
from accelerate import Accelerator
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports using relative paths
from .model import get_model
from .utils import get_hyperparameters, save_experiment_results, plot_training_curves, REPORT_HYPERPARAMETERS
from .train import run_training_config

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()

# The AblationExperiment class is removed.

class ECANetAblation:
    """ECA-Net消融实验配置.
    
    Ablations from report section 5.1:
    1. Baseline (ResNet-20 no ECA)
    2. ECA-ResNet-20 (adaptive k_size)
    3. ECA-ResNet-20 (k_size=3)
    4. ECA-ResNet-20 (k_size=5)
    5. ECA-ResNet-20 (k_size=7)
    6. ECA-ResNet-20 (k_size=9)
    """
    
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca' 

        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20_No_ECA_Baseline'
        })
        configs.append({
            'model_name': 'ecanet20_adaptive',
            'config_override': {},
            'label': 'ECANet-20_Adaptive_k_size'
        })
        configs.append({
            'model_name': 'ecanet20_fixed_k3', 
            'config_override': {},
            'label': 'ECANet-20_Fixed_k_size_3'
        })
        configs.append({
            'model_name': 'ecanet20_fixed_k5',
            'config_override': {},
            'label': 'ECANet-20_Fixed_k_size_5'
        })
        configs.append({
            'model_name': 'ecanet20_fixed_k7',
            'config_override': {},
            'label': 'ECANet-20_Fixed_k_size_7'
        })
        configs.append({
            'model_name': 'ecanet20_fixed_k9',
            'config_override': {},
            'label': 'ECANet-20_Fixed_k_size_9'
        })
        return configs

class GhostNetAblation:
    """GhostNet消融实验配置 (Report Section 5.2).
    1. Baseline (ResNet-20 no ECA)
    2. Ghost-ResNet-20 (ratio=2)
    3. Ghost-ResNet-20 (ratio=3)
    4. Ghost-ResNet-20 (ratio=4)
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca' 
        ghost_model_name = 'ghost_resnet_20'

        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20_Baseline_for_GhostNet'
        })
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 2}},
            'label': 'Ghost-ResNet-20_ratio_2'
        })
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 3}},
            'label': 'Ghost-ResNet-20_ratio_3'
        })
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 4}},
            'label': 'Ghost-ResNet-20_ratio_4'
        })
        return configs


# Note: Attention Position Ablation is more complex as it requires different model block definitions.
# This is not easily configurable via hyperparameter overrides unless model.py supports such variants.
# For now, we will skip implementing the AttentionPositionAblation runner that requires code changes in model.py.

class AttentionPositionAblation:
    """注意力模块位置消融实验 (报告5.3节).
    1. Baseline (ResNet-20 no ECA)
    2. ECA after first Conv (Pos1, k_size=3)
    3. ECA after second Conv before Add (Pos2 - default eca_resnet_20, k_size=3)
    4. ECA after Add on residual (Pos3, k_size=3)
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca'
        default_k_size = 3 

        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20_No_ECA_Baseline_for_Position'
        })
        configs.append({
            'model_name': 'eca_resnet20_pos1',
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': f'ECA-ResNet20_Pos1_k{default_k_size}'
        })
        configs.append({
            'model_name': 'eca_resnet_20', 
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': f'ECA-ResNet20_Pos2_Default_k{default_k_size}'
        })
        configs.append({
            'model_name': 'eca_resnet20_pos3',
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': f'ECA-ResNet20_Pos3_k{default_k_size}'
        })
        return configs

def run_ablation_study(ablation_configs, study_name):
    print(f"\n\n{'='*80}")
    print(f"🔬 Starting Ablation Study: {study_name} 🔬")
    print(f"{'='*80}")

    for config in ablation_configs:
        model_name = config['model_name']
        config_override_from_ablation = config.get('config_override', {}).copy()
        label = config.get('label', model_name) 

        # 为确保日志目录清晰，使用 study_name 和 label 生成 run_name
        # 清理 label 中的特殊字符，使其适合作为目录名
        sanitized_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("=", "").replace(",", "").replace("/", "_").replace("-", "_")
        
        effective_run_name = f"{study_name}_{sanitized_label}"

        # Prepare the programmatic_config_override for run_training_config
        # This dictionary will be passed to TrainingArguments and the model constructor
        programmatic_override = config_override_from_ablation
        programmatic_override['run_name'] = effective_run_name # Ensure TrainingArguments uses this for output dirs

        # 统一设置消融实验的训练轮数为300
        programmatic_override['num_train_epochs'] = 300

        print(f"--- Running Ablation Case: {label} (Model: {model_name}) ---")
        print(f"    Effective Run Name for logs: {effective_run_name}")
        print(f"    Config Overrides: {programmatic_override}")
        
        try:
            run_training_config(model_name=model_name, programmatic_config_override=programmatic_override)
            print(f"--- Ablation Case: {label} (Model: {model_name}) completed successfully. ---")
        except Exception as e:
            print(f"ERROR during ablation case: {label} (Model: {model_name})")
            print(f"Error details: {e}")
            # Potentially log this error to a file or re-raise if one failure should stop all
        
        # Optional: Add a small delay or resource check if running many GPU-intensive jobs sequentially
        # time.sleep(5) 

    print(f"\n---- Ablation Study: {study_name} Complete ----")
    print("Individual model training results should be saved in subdirectories under logs/results.")
    print("Run analyze_results.py to aggregate and analyze these results.")

def run_all_ablation_experiments():
    print("Starting All Ablation Experiments Orchestration Script")
    # 创建必要的目录，Trainer中的逻辑可能也会创建，这里确保它们存在
    Path("logs/results").mkdir(parents=True, exist_ok=True)
    Path("logs/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("assets").mkdir(parents=True, exist_ok=True)

    eca_configs = ECANetAblation.get_experiment_configs()
    run_ablation_study(eca_configs, study_name="ECA_Net_Ablation")

    ghost_configs = GhostNetAblation.get_experiment_configs()
    run_ablation_study(ghost_configs, study_name="GhostNet_Ablation")
    
    attention_pos_configs = AttentionPositionAblation.get_experiment_configs()
    run_ablation_study(attention_pos_configs, study_name="Attention_Position_Ablation")

    print("\nAll ablation studies attempted.")

if __name__ == '__main__':
    # 这里的 Accelerator 初始化仅用于可能的顶层分布式脚本控制，
    # 但由于 Trainer 现在处理自己的 Accelerator，可能不需要在这里显式管理 Accelerator 实例。
    # 如果脚本本身不需要分布式控制（例如，它只在主进程上编排单次训练），则不需要 Accelerator。
    # accelerator = Accelerator()
    # if accelerator.is_main_process: # 只在主进程运行编排逻辑
    #    run_all_ablation_experiments()
    # else:
    #    # 在非主进程中，如果 Trainer 需要所有进程都启动，则可能需要某种同步
    #    # 但通常 Trainer 的启动方式会处理这个问题。
    #    pass 
    # 简化：假设此脚本由单个进程运行以启动多个（可能是分布式的）训练作业
    run_all_ablation_experiments() 