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

# Project imports
from model import get_model # Relies on the new get_model
from utils import get_hyperparameters, save_experiment_results, plot_training_curves, REPORT_HYPERPARAMETERS
from train_all_models import run_training_for_model # Key import

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
        base_model_name = 'resnet20_no_eca' # Explicit baseline

        # 1. Baseline ResNet-20 (No ECA)
        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20 (No ECA Baseline)'
        })

        # 2. ECA-Net with adaptive k_size
        configs.append({
            'model_name': 'ecanet20_adaptive',
            'config_override': {}, # Adaptive k_size is inherent to the model block
            'label': 'ECANet-20 (Adaptive k_size)'
        })

        # 3. ECA-Net with fixed k_size=3
        configs.append({
            'model_name': 'ecanet20_fixed_k3', 
            'config_override': {}, # k_size=3 is inherent to the model block
            'label': 'ECANet-20 (Fixed k_size=3)'
        })

        # 4. ECA-Net with fixed k_size=5
        configs.append({
            'model_name': 'ecanet20_fixed_k5',
            'config_override': {}, # k_size=5 is inherent to the model block
            'label': 'ECANet-20 (Fixed k_size=5)'
        })
        
        # 5. ECA-Net with fixed k_size=7
        configs.append({
            'model_name': 'ecanet20_fixed_k7',
            'config_override': {}, # k_size=7 is inherent to the model block
            'label': 'ECANet-20 (Fixed k_size=7)'
        })

        # 6. ECA-Net with fixed k_size=9
        configs.append({
            'model_name': 'ecanet20_fixed_k9',
            'config_override': {}, # k_size=9 is inherent to the model block
            'label': 'ECANet-20 (Fixed k_size=9)'
        })
        return configs

class GhostNetAblation:
    """GhostNet消融实验配置 (Report Section 5.2).
    1. Baseline (ResNet-20 no ECA)
    2. Ghost-ResNet-20 (ratio=2, default for ghost_resnet_20)
    3. Ghost-ResNet-20 (ratio=3)
    4. Ghost-ResNet-20 (ratio=4)
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca' # Explicit baseline
        ghost_model_name = 'ghost_resnet_20'

        # 1. Baseline ResNet-20 (No ECA, equivalent to standard ResNet-20)
        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20 (Baseline for GhostNet Ablation)'
        })

        # 2. Ghost-ResNet-20 with ratio=2 (default from get_hyperparameters for ghost_resnet_20)
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 2}}, # Explicit
            'label': 'Ghost-ResNet-20 (ratio=2)'
        })

        # 3. Ghost-ResNet-20 with ratio=3
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 3}},
            'label': 'Ghost-ResNet-20 (ratio=3)'
        })
        
        # 4. Ghost-ResNet-20 with ratio=4
        configs.append({
            'model_name': ghost_model_name,
            'config_override': {'model_constructor_params': {'ratio': 4}},
            'label': 'Ghost-ResNet-20 (ratio=4)'
        })
        return configs


# Note: Attention Position Ablation is more complex as it requires different model block definitions.
# This is not easily configurable via hyperparameter overrides unless model.py supports such variants.
# For now, we will skip implementing the AttentionPositionAblation runner that requires code changes in model.py.

class AttentionPositionAblation:
    """注意力模块位置消融实验 (报告5.3节).
    1. Baseline (ResNet-20 no ECA)
    2. ECA after first Conv (Pos1)
    3. ECA after second Conv before Add (Pos2 - default eca_resnet_20)
    4. ECA after Add on residual (Pos3)
    All ECA variants use k_size=3 for this position ablation, based on report/common practice.
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca'
        default_k_size = 3 # As per report/common choice for position ablation

        # 1. Baseline ResNet-20 (No ECA)
        configs.append({
            'model_name': base_model_name,
            'config_override': {},
            'label': 'ResNet-20 (No ECA Baseline for Position Ablation)'
        })

        # 2. ECA after first Conv (Pos1)
        configs.append({
            'model_name': 'eca_resnet20_pos1',
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': 'ECA-ResNet20 (Pos1: After Conv1)'
        })
        
        # 3. ECA after second Conv, before Add (Pos2 - standard ECABasicBlock)
        # This uses the 'eca_resnet_20' builder which defaults to ECABasicBlock
        configs.append({
            'model_name': 'eca_resnet_20', 
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': 'ECA-ResNet20 (Pos2: After Conv2, Before Add)'
        })

        # 4. ECA after Add on residual (Pos3)
        configs.append({
            'model_name': 'eca_resnet20_pos3',
            'config_override': {'model_constructor_params': {'k_size': default_k_size}},
            'label': 'ECA-ResNet20 (Pos3: After Add)'
        })
        return configs

def run_ablation_study(ablation_configs, accelerator, study_name):
    if accelerator.is_local_main_process:
        print(f"\n\n{'='*80}")
        print(f"🔬 Starting Ablation Study: {study_name} 🔬")
        print(f"{'='*80}")

    for config in ablation_configs:
        model_name = config['model_name']
        config_override = config.get('config_override', {})
        label = config.get('label', model_name) # Label can be used for logging/distinguishing runs

        if accelerator.is_local_main_process:
            print(f"--- Running Ablation Case: {label} (Model: {model_name}, Override: {config_override}) ---")
        
        # To make output files unique for ablation runs on the same base model name,
        # we should ideally include the label or a summary of the override in the output name.
        # For now, run_training_for_model uses the base model_name. The hparams in the JSON will differ.
        # Consider modifying run_training_for_model to accept an optional `run_name_suffix` for output files.
        # As a temporary workaround if needed, or if generate_results.py can handle it:
        # The `label` or a hash of `config_override` could be part of what `run_training_for_model` logs
        # or how `save_experiment_results` names files if we customize it further.
        
        # For now, rely on the config_override being logged inside the JSON for differentiation.
        # The filename will be based on `model_name`.
        run_training_for_model(model_name, accelerator, config_override=config_override)

    if accelerator.is_local_main_process:
        print(f"\n---- Ablation Study: {study_name} Complete ----")
        print("Individual model training results saved in logs/results.")
        print("Run generate_results.py to analyze these results.")

def run_all_ablation_experiments():
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Starting All Ablation Experiments Orchestration Script")
        Path("logs/results").mkdir(parents=True, exist_ok=True)
        Path("logs/checkpoints").mkdir(parents=True, exist_ok=True)
        Path("assets").mkdir(parents=True, exist_ok=True)

    # ECA-Net Ablations
    eca_configs = ECANetAblation.get_experiment_configs()
    run_ablation_study(eca_configs, accelerator, "ECA-Net k_size Ablation")
    accelerator.wait_for_everyone()

    # GhostNet Ablations
    ghost_configs = GhostNetAblation.get_experiment_configs()
    run_ablation_study(ghost_configs, accelerator, "GhostNet Ratio Ablation")
    accelerator.wait_for_everyone()
    
    # Attention Position Ablations
    attn_pos_configs = AttentionPositionAblation.get_experiment_configs()
    run_ablation_study(attn_pos_configs, accelerator, "ECA Attention Position Ablation")
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        print("\nAll ablation experiments orchestration finished.")
        print("Individual results are in logs/results/. Run generate_results.py for summaries.")

if __name__ == "__main__":
    run_all_ablation_experiments() 