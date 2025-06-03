#!/usr/bin/env python3 # 保持shebang，虽然-m运行时可能不直接用
#!/usr/bin/env python3 # 保持shebang，虽然-m运行时可能不直接用
import torch
# import torch.nn as nn # No longer directly used here
# import torch.optim as optim # No longer directly used here
# from torch.utils.data import DataLoader # No longer directly used here
# import torchvision # No longer directly used here
# import torchvision.transforms as transforms # No longer directly used here
# from accelerate import Accelerator # Removed
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess # Added for subprocess calls
import sys # Added
import os # Added for environment variables
# import os # Added # Removed since no longer needed
# import torch.distributed as dist # Added # Removed since no longer needed

# Project imports using relative paths
from .model import get_model
from .utils import get_hyperparameters, save_experiment_results, plot_training_curves, REPORT_HYPERPARAMETERS
from .train import run_training_config
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
    2. Ghost-ResNet-20 (ratio=2)
    3. Ghost-ResNet-20 (ratio=3)
    4. Ghost-ResNet-20 (ratio=4)
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca' 
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
    2. ECA after first Conv (Pos1, k_size=3)
    3. ECA after second Conv before Add (Pos2 - default eca_resnet_20, k_size=3)
    4. ECA after Add on residual (Pos3, k_size=3)
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'resnet20_no_eca'
        default_k_size = 3 
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

class ImprovedResNetConvNeXtAblation:
    """Improved-ResNet20-ConvNeXt 相关模型的消融实验.
    1. Baseline (improved_resnet20_convnext)
    2. No DropPath
    3. Standard 3x3 Conv (instead of 7x7 depthwise)
    4. No Inverted Bottleneck
    """
    @staticmethod
    def get_experiment_configs():
        configs = []
        base_model_name = 'improved_resnet20_convnext'

        # Baseline - the original improved_resnet20_convnext
        configs.append({
            'model_name': base_model_name,
            'config_override': {}, # Assuming default drop_path_rate=0.05 from its definition
            'label': 'ImprovedResNet20ConvNeXt_Baseline'
        })
        
        # No DropPath variant
        configs.append({
            'model_name': 'improved_resnet20_convnext_no_droppath',
            'config_override': {}, # drop_path_rate=0.0 is set in its builder
            'label': 'ImprovedResNet20ConvNeXt_NoDropPath'
        })

        # Standard 3x3 Conv variant
        configs.append({
            'model_name': 'improved_resnet20_convnext_std_conv',
            'config_override': {}, # drop_path_rate=0.05 is set in its builder
            'label': 'ImprovedResNet20ConvNeXt_StdConv'
        })

        # No Inverted Bottleneck variant
        configs.append({
            'model_name': 'improved_resnet20_convnext_no_inverted',
            'config_override': {}, # drop_path_rate=0.05 is set in its builder
            'label': 'ImprovedResNet20ConvNeXt_NoInvertedBottleneck'
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

        # 检查是否已有实验结果
        # Trainer 将结果保存在 args.output_dir / model_name / args.run_name
        # args.output_dir 可能在 programmatic_override 中被覆盖，否则默认为 ./logs (由TrainingArguments定义)
        base_output_dir = programmatic_override.get('output_dir', './logs')
        
        # Trainer 实际的保存路径包含模型名称和运行名称两层目录
        # 参考 trainer.py 中的 effective_output_dir 构建逻辑
        expected_log_dir = Path(base_output_dir) / model_name / effective_run_name
        summary_file_path = expected_log_dir / "evaluation_summary.json"

        if summary_file_path.exists():
            print(f"--- Skipping Ablation Case: {label} (Model: {model_name}) ---")
            print(f"    Results (evaluation_summary.json) already found at: {summary_file_path}")
            continue # 跳到下一个配置

        print(f"--- Running Ablation Case: {label} (Model: {model_name}) ---")
        print(f"    Effective Run Name for logs: {effective_run_name}")
        print(f"    Config Overrides: {programmatic_override}")
        
        try:
            # 使用 torchrun 启动训练任务以获得最佳多GPU性能
            # 构建命令行参数
            cmd = [
                "torchrun", 
                "--nproc_per_node=auto",  # 自动检测GPU数量
                "-m", "src.train",
                "--model_name", model_name
            ]
            
            # 添加配置覆盖参数
            if 'run_name' in programmatic_override:
                cmd.extend(["--run_name", programmatic_override['run_name']])
            if 'num_train_epochs' in programmatic_override:
                cmd.extend(["--epochs", str(programmatic_override['num_train_epochs'])])
            if 'output_dir' in programmatic_override:
                cmd.extend(["--output_dir", programmatic_override['output_dir']])
            
            # 设置环境变量用于传递model_constructor_params
            env = {}
            if 'model_constructor_params' in programmatic_override:
                env['MODEL_CONSTRUCTOR_PARAMS'] = json.dumps(programmatic_override['model_constructor_params'])
            
            print(f"    执行命令: {' '.join(cmd)}")
            if env:
                print(f"    环境变量: {env}")
            
            # 执行命令，不捕获输出以便实时查看训练进度
            result = subprocess.run(
                cmd, 
                timeout=7200,  # 2小时超时
                env={**os.environ, **env} if env else None
            )
            
            if result.returncode == 0:
                print(f"--- Ablation Case: {label} (Model: {model_name}) completed successfully. ---")
            else:
                print(f"ERROR during ablation case: {label} (Model: {model_name})")
                print(f"Return code: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT during ablation case: {label} (Model: {model_name}) - exceeded 2 hours")
        except Exception as e:
            print(f"ERROR during ablation case: {label} (Model: {model_name})")
            print(f"Error details: {e}")
            # Potentially log this error to a file or re-raise if one failure should stop all
        
        # Optional: Add a small delay or resource check if running many GPU-intensive jobs sequentially
        # time.sleep(5) 

    print(f"\n---- Ablation Study: {study_name} Complete ----")
    print("Run analyze_results.py to aggregate and analyze these results.")

def run_all_ablation_experiments():
    print("Starting All Ablation Experiments Orchestration Script")
    
    # eca_configs = ECANetAblation.get_experiment_configs()
    # run_ablation_study(eca_configs, study_name="ECA_Net_Ablation")

    # ghost_configs = GhostNetAblation.get_experiment_configs()
    # run_ablation_study(ghost_configs, study_name="GhostNet_Ablation")
    
    attention_pos_configs = AttentionPositionAblation.get_experiment_configs()
    run_ablation_study(attention_pos_configs, study_name="Attention_Position_Ablation")

    improved_resnet_convnext_configs = ImprovedResNetConvNeXtAblation.get_experiment_configs()
    run_ablation_study(improved_resnet_convnext_configs, study_name="Improved_ResNet_ConvNeXt_Ablation")

    print("\nAll ablation studies attempted.")

if __name__ == '__main__':
    # 现在脚本作为单进程运行，直接执行消融实验
    # 每个具体的训练任务会在内部根据可用的GPU资源自动处理分布式训练
    run_all_ablation_experiments() 