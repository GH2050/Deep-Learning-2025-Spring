import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
import pandas as pd
import logging
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Hyperparameter Configuration (from Report Section 3.4.3) ---
# Note: Batch sizes here are per GPU (V100 16GB). Total batch size will be this * num_gpus.
# All models trained for 200 epochs.

REPORT_HYPERPARAMETERS = {
    'epochs': 200,
    'categories': {
        'ResNet_Variants': { # resnet_20, resnet_32, resnet_56, eca_resnet_20, eca_resnet_32
            'optimizer_type': 'sgd',
            'lr': 0.1,
            'scheduler_type': 'cosine_annealing',
            'warmup_epochs': 0,
            'weight_decay': 5e-4,
            'batch_size_per_gpu': 128,
            'use_imagenet_norm': False, # CIFAR-100 specific normalization
            'model_specific_params': {
                'eca_resnet_20': {'k_size': 3},
                'eca_resnet_32': {'k_size': 5},
                #### ECA-NET 20 variants
                'ecanet20_adaptive': {},  # 自适应模型不需要特定的 k_size
                'ecanet20_fixed_k3': {'k_size': 3},  # 固定 k=3
                'resnet20_no_eca': {},  # 无 ECA 模型不需要特定参数
                'ecanet20_fixed_k5': {'k_size': 5},  # 固定 k=5
                'ecanet20_fixed_k7': {'k_size': 7},  # 固定 k=7
                'ecanet20_fixed_k9': {'k_size': 9},  # 固定 k=9
                #### ResNet 20 variants
            }
        },
        'GhostNet_Variants': { # ghost_resnet_20, ghost_resnet_32, ghostnet_100 (custom)
            'optimizer_type': 'sgd',
            'lr': 0.1,
            'scheduler_type': 'cosine_annealing',
            'warmup_epochs': 5,
            'weight_decay': 4e-5,
            'batch_size_per_gpu': 128,
            'use_imagenet_norm': False, # CIFAR-100 specific normalization
            'model_specific_params': {
                'ghost_resnet_20': {'ratio': 2},
                'ghost_resnet_32': {'ratio': 2},
                'ghostnet_100': {'batch_size_per_gpu': 64} # ghostnet_100 is larger
            }
        },
        'ConvNeXt_Tiny': { # convnext_tiny (custom)
            'optimizer_type': 'adamw',
            'lr': 4e-3, # From report for CIFAR. Original paper: base_lr (5e-4) * total_batch_size / 1024
            'scheduler_type': 'cosine_annealing_warmup',
            'warmup_epochs': 20,
            'weight_decay': 0.05,
            'batch_size_per_gpu': 64,
            'use_imagenet_norm': False, # CIFAR-100 specific normalization. ConvNeXt uses LayerNorm.
            'model_specific_params': {}
        },
        'Hybrid_Attention_CNN': { # coatnet_0_custom, cspresnet50, hornet_tiny_custom, resnest50d, coatnet_0_custom_enhanced
            'optimizer_type': 'adamw',
            'lr': 2e-4,
            'scheduler_type': 'cosine_annealing_warmup',
            'warmup_epochs': 10,
            'weight_decay': 0.05,
            'batch_size_per_gpu': 64,
            'use_imagenet_norm': False, # CIFAR-100 specific normalization
            'model_specific_params': {
                'coatnet_0_custom_enhanced': { # Default LSK parameters for the enhanced model
                    'batch_size_per_gpu': 64,
                    'lsk_kernel_sizes': [3, 5, 7], 
                    'lsk_reduction_ratio': 8,
                    'se_ratio_in_mbconv': 0.25
                } 
            }
        },
        'MLP_Mixer_Variants': { # mlp_mixer_tiny (custom), mlp_mixer_b16 (custom)
            'optimizer_type': 'adamw',
            'lr': 1e-3,
            'scheduler_type': 'cosine_annealing_warmup',
            'warmup_epochs': 10,
            'weight_decay': 0.05,
            'batch_size_per_gpu': 128, # Mixers can be efficient
            'use_imagenet_norm': False, # CIFAR-100 specific normalization
            'model_specific_params': {
                 'mlp_mixer_b16': {'batch_size_per_gpu': 64} # B/16 custom is larger
            }
        },
        'SegNeXt_MSCAN_Tiny': { # segnext_mscan_tiny (custom)
            'optimizer_type': 'adamw',
            'lr': 1e-3,
            'scheduler_type': 'cosine_annealing_warmup',
            'warmup_epochs': 10,
            'weight_decay': 0.05,
            'batch_size_per_gpu': 128,
            'use_imagenet_norm': False, # CIFAR-100 specific normalization
            'model_specific_params': {}
        }
    },
    'model_to_category': {
        'resnet_20': 'ResNet_Variants',
        'resnet_32': 'ResNet_Variants',
        'resnet_56': 'ResNet_Variants',
        'eca_resnet_20': 'ResNet_Variants',
        'eca_resnet_32': 'ResNet_Variants',
        'ghost_resnet_20': 'GhostNet_Variants',
        'ghost_resnet_32': 'GhostNet_Variants',
        'ghostnet_100': 'GhostNet_Variants',      # Was ghostnet_100 (timm)
        'convnext_tiny': 'ConvNeXt_Tiny',        # Was convnext_tiny (custom) and convnext_tiny_timm
        'segnext_mscan_tiny': 'SegNeXt_MSCAN_Tiny',
        'coatnet_0_custom': 'Hybrid_Attention_CNN', # Was coatnet_0 (timm)
        'cspresnet50': 'Hybrid_Attention_CNN',       # Was cspresnet50 (timm)
        'hornet_tiny_custom': 'Hybrid_Attention_CNN',# Was hornet_tiny (timm)
        'resnest50d': 'Hybrid_Attention_CNN',        # Was resnest50d (timm)
        'mlp_mixer_tiny': 'MLP_Mixer_Variants',
        'mlp_mixer_b16': 'MLP_Mixer_Variants',      # Was mlp_mixer_b16 (timm)
        'coatnet_0_custom_enhanced': 'Hybrid_Attention_CNN', # Added new model to category
        'coatnet_cifar_opt': 'Hybrid_Attention_CNN',         # Added for the new CIFAR-optimized CoAtNet
        'coatnet_cifar_opt_large_stem': 'Hybrid_Attention_CNN', # Added for the large stem variant
        'ecanet20_adaptive': 'ResNet_Variants', #### ECA-NET 20 adaptive
        'ecanet20_fixed_k3': 'ResNet_Variants', #### ECA-NET 20 fixed k=3
        'resnet20_no_eca': 'ResNet_Variants', #### ResNet 20 without ECA
        'ecanet20_fixed_k5': 'ResNet_Variants', #### ECA-NET 20 fixed k=5
        'ecanet20_fixed_k7': 'ResNet_Variants', #### ECA-NET 20 fixed k=7
        'ecanet20_fixed_k9': 'ResNet_Variants', #### ECA-NET 20 fixed k=9

    }
}

def get_hyperparameters(model_name: str):
    """
    Retrieves hyperparameters for a given model based on REPORT_HYPERPARAMETERS.
    Args:
        model_name (str): The name of the model.
    Returns:
        dict: A dictionary containing hyperparameters.
    """
    # Handle potential legacy timm model names if they are passed by mistake
    # This mapping should align with the logic in model.py's get_model
    legacy_map = {
        "convnext_tiny_timm": "convnext_tiny",
        "ghostnet_100_timm": "ghostnet_100",
        "mlp_mixer_b16_timm": "mlp_mixer_b16",
        "coatnet_0": "coatnet_0_custom", # Direct map, coatnet_0 was the timm key
        "cspresnet50_timm": "cspresnet50",
        "hornet_tiny_timm": "hornet_tiny_custom",
        "resnest50d_timm": "resnest50d"
    }
    if model_name in legacy_map:
        # print(f"Hyperparameter lookup: mapping legacy model name '{model_name}' to '{legacy_map[model_name]}'")
        model_name = legacy_map[model_name]

    category_name = REPORT_HYPERPARAMETERS['model_to_category'].get(model_name)
    if not category_name:
        # Try to infer category if a _custom or similar suffix was missed in mapping
        # This is a fallback, explicit mapping is better.
        for key_pattern in ['_custom', '_timm']: # Check if removing a suffix helps
            if model_name.endswith(key_pattern):
                base_name = model_name[:-len(key_pattern)]
                category_name = REPORT_HYPERPARAMETERS['model_to_category'].get(base_name)
                if category_name:
                    # print(f"Hyperparameter lookup: found category for base name '{base_name}' for model '{model_name}'")
                    model_name = base_name # Use the base name for specific params if found this way
                    break
    
    if not category_name:
        raise ValueError(f"Model {model_name} not found in model_to_category mapping after attempting fallbacks.")

    category_hparams = REPORT_HYPERPARAMETERS['categories'][category_name].copy()
    # Get model specific params using the potentially updated model_name
    model_specific_hparams = category_hparams.get('model_specific_params', {}).get(model_name, {})
    
    final_hparams = {
        'optimizer_type': category_hparams['optimizer_type'],
        'lr': category_hparams['lr'],
        'scheduler_type': category_hparams['scheduler_type'],
        'warmup_epochs': category_hparams['warmup_epochs'],
        'weight_decay': category_hparams['weight_decay'],
        'batch_size_per_gpu': category_hparams['batch_size_per_gpu'],
        'use_imagenet_norm': category_hparams['use_imagenet_norm'], # This will now be False for all
        'epochs': REPORT_HYPERPARAMETERS['epochs'],
        'model_name': model_name # Store the final model name used for lookup
    }

    final_hparams.update(model_specific_hparams) # Override with specific settings for the model

    model_constructor_params = {}
    # Extract known model constructor params from the final hparams
    # These keys should match what model_specific_params might contain for model builders
    known_constructor_keys = ['k_size', 'ratio', 'width', 'dropout', 
                              'depths', 'dims', # For ConvNeXt, MSCAN
                              'image_size', 'patch_size', 'dim', 'depth', 'token_mlp_dim', 'channel_mlp_dim', # For MLP Mixer
                              'radix', 'cardinality', 'bottleneck_width', 'deep_stem', 'stem_width', 'avg_down', 'avd', 'avd_first', # For ResNeSt
                              # For CoAtNetCustom (simplified)
                              's0_channels', 's1_channels', 's2_channels', 's3_channels', 's4_channels',
                              's0_blocks', 's1_blocks', 's2_blocks', 's3_blocks', 's4_blocks',
                              'mbconv_expand_ratio', 'transformer_heads', 'transformer_mlp_dim_ratio',
                              # For HorNetCustom (simplified)
                              'order', 'dw_k', 'use_filter',
                              # For LSKNet components in enhanced models
                              'lsk_kernel_sizes', 'lsk_reduction_ratio', 'se_ratio_in_mbconv'
                             ] 
    for key in known_constructor_keys:
        if key in final_hparams: # Check if it was set directly by model_specific_params
            model_constructor_params[key] = final_hparams[key]
        elif key in category_hparams.get('model_specific_params', {}).get(model_name, {}): # Check original model_specific_params
             model_constructor_params[key] = category_hparams['model_specific_params'][model_name][key]


    final_hparams['model_constructor_params'] = model_constructor_params
    
    # Ensure 'use_imagenet_norm' is consistently False as we are not using TIMM pretrained models
    final_hparams['use_imagenet_norm'] = False

    return final_hparams

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, model_name="Model"):
    total_params = count_parameters(model)
    print(f'{model_name} 总参数量: {total_params:,} ({total_params/1e6:.2f}M)')
    # Dummy input test removed, should be done in model testing or training script

def plot_training_curves(history, title_prefix, output_dir='assets', loss_filename="training_loss.png", accuracy_filename="training_accuracy.png"):
    # The function will still save one combined image, we'll use accuracy_filename for the output file.
    # If separate files are strictly needed, this function requires more significant refactoring.
    save_path = os.path.join(output_dir, accuracy_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} 训练曲线', fontsize=16)
    
    epochs_range = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs_range, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs_range, history['test_losses'], 'r-', label='测试损失')
    ax1.set_title('损失函数')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_range, history['train_accs'], 'b-', label='训练准确率 Top-1')
    ax2.plot(epochs_range, history['test_accs'], 'r-', label='测试准确率 Top-1')
    ax2.set_title('准确率 Top-1')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    last_n = min(20, len(epochs_range))
    ax3.plot(epochs_range[-last_n:], history['train_losses'][-last_n:], 'b-', label='训练损失')
    ax3.plot(epochs_range[-last_n:], history['test_losses'][-last_n:], 'r-', label='测试损失')
    ax3.set_title(f'损失函数（最后{last_n}轮）')
    ax3.set_xlabel('轮次')
    ax3.set_ylabel('损失')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(epochs_range[-last_n:], history['train_accs'][-last_n:], 'b-', label='训练准确率 Top-1')
    ax4.plot(epochs_range[-last_n:], history['test_accs'][-last_n:], 'r-', label='测试准确率 Top-1')
    ax4.set_title(f'准确率 Top-1（最后{last_n}轮）')
    ax4.set_xlabel('轮次')
    ax4.set_ylabel('准确率 (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'训练曲线图已保存到: {save_path}')

def create_markdown_table(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a Markdown table string."""
    return df.to_markdown(index=False)

def save_experiment_results(
    results_data: dict,
    model_name: str, # This is the base model_name from args
    hparams: dict,
    output_dir: str = 'logs/results',
    metrics_history: dict = None, # e.g., train_losses, test_losses, train_accs, test_accs
    run_label: str = None, # New parameter for specific run identification (e.g., from ablation)
    filename: str = "experiment_results.json" # New parameter for the output filename
    ):
    """
    Saves experiment results, hyperparameters, and metrics history to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the provided filename directly
    save_path = os.path.join(output_dir, filename)
    
    data_to_save = {
        'model_name': model_name, # Base model name
        'run_label': run_label if run_label else model_name, # Effective unique identifier for this run
        'results': results_data, 
        'hyperparameters': hparams,
        'metrics_history': metrics_history if metrics_history else {},
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f'详细实验结果已保存到: {save_path}')
    except IOError as e:
        print(f'保存结果到 {save_path} 失败: {e}')
    except Exception as e:
        print(f'序列化结果到JSON时发生错误: {e}')

def compare_models(results_dict, save_path='logs/model_comparison.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    models = list(results_dict.keys())
    # Assuming results_dict[model] has {'results': {'best_acc': X, 'test_accs': [...]}}
    best_accs = [results_dict[model]['results'].get('best_test_acc_top1', 0) for model in models]
    final_accs = [results_dict[model]['metrics_history'].get('test_accs', [0])[-1] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('模型性能对比 (Top-1准确率)', fontsize=16)
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, best_accs, width, label='最佳准确率', alpha=0.8, color='skyblue')
    rects2 = ax1.bar(x + width/2, final_accs, width, label='最终准确率', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('最佳 vs 最终测试准确率', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(max(best_accs), max(final_accs)) * 1.1 + 5) # Adjust y-lim dynamically

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    for model in models:
        test_accs_history = results_dict[model]['metrics_history'].get('test_accs', [])
        if test_accs_history:
            epochs_range = range(1, len(test_accs_history) + 1)
            ax2.plot(epochs_range, test_accs_history, label=model, linewidth=2, marker='.', markersize=5)
    
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('测试准确率 (%)', fontsize=12)
    ax2.set_title('测试准确率变化曲线', fontsize=14)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(min(final_accs)-10 if final_accs else 0, max(best_accs)+5 if best_accs else 100) # Dynamic y-lim
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'模型对比图已保存到: {save_path}')

def save_comparison_summary_text(results_dict, save_path='assets/comparison_summary.txt'):
    # Renamed from save_comparison_results to avoid conflict if old one exists, and changed save path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('='*80 + '\n')
        f.write('CIFAR-100 分类模型性能对比总结 (基于 *_results.json 文件)\n')
        f.write('='*80 + '\n\n')
        
        header = f'{"模型名称":<25} {"最佳Top-1 (%)":<15} {"最终Top-1 (%)":<15} {"参数量 (M)":<15} {"训练时间 (h)":<15}'
        f.write(header + '\n')
        f.write('-'*len(header) + '\n')
        
        sorted_models = sorted(results_dict.keys(), key=lambda m: results_dict[m]['results'].get('best_test_acc_top1', 0), reverse=True)

        for model_name in sorted_models:
            res_data = results_dict[model_name]['results']
            metrics_hist = results_dict[model_name]['metrics_history']
            
            best_acc = res_data.get('best_test_acc_top1', 0)
            final_acc = metrics_hist.get('test_accs', [0])[-1]
            params_m = res_data.get('params_M', 0)
            train_time_s = res_data.get('train_time_total_seconds', 0)
            train_time_h = train_time_s / 3600.0 if train_time_s else 0

            f.write(f'{model_name:<25} {best_acc:<15.2f} {final_acc:<15.2f} {params_m:<15.2f} {train_time_h:<15.2f}\n')
        
        f.write('\n' + '='*80 + '\n')
        
        if sorted_models:
            best_overall_model = sorted_models[0]
            best_overall_acc = results_dict[best_overall_model]['results'].get('best_test_acc_top1', 0)
            f.write(f'综合最佳模型: {best_overall_model} (准确率: {best_overall_acc:.2f}%)\n')
        
        baseline_model_name = 'resnet_20' # Or choose a more appropriate baseline
        baseline_acc = results_dict.get(baseline_model_name, {}).get('results', {}).get('best_test_acc_top1', 0)

        if baseline_acc > 0:
            f.write(f'\n相对于基线 {baseline_model_name} ({baseline_acc:.2f}%) 的提升:\n')
            for model_name in sorted_models:
                if model_name != baseline_model_name:
                    improvement = results_dict[model_name]['results'].get('best_test_acc_top1', 0) - baseline_acc
                    f.write(f'{model_name:<25}: {improvement:+.2f}%\n')
    
    print(f'对比总结文本已保存到: {save_path}')

def setup_logging(save_dir, rank=0):
    """设置日志记录"""
    if rank == 0:
        log_file = os.path.join(save_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("="*50)
        logger.info("CIFAR-100 训练开始")
        logger.info("="*50)
        return logger
    return None

def log_system_info(logger, rank=0):
    """记录系统信息"""
    if rank == 0 and logger:
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前设备: {torch.cuda.current_device()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('分布式环境变量未设置，使用单GPU训练')
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return True, rank, world_size, gpu

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_optimizer_scheduler(model, hparams, steps_per_epoch):
    """获取优化器和学习率调度器"""
    if hparams['optimizer_type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=hparams['lr'], 
            momentum=0.9, 
            weight_decay=hparams['weight_decay'],
            nesterov=True
        )
    elif hparams['optimizer_type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay']
        )
    else:
        raise ValueError(f"不支持的优化器类型: {hparams['optimizer_type']}")
    
    if hparams['scheduler_type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=hparams['epochs'],
            eta_min=1e-6
        )
    elif hparams['scheduler_type'] == 'cosine_annealing_warmup':
        def lr_lambda(epoch):
            if epoch < hparams['warmup_epochs']:
                return epoch / hparams['warmup_epochs']
            else:
                progress = (epoch - hparams['warmup_epochs']) / (hparams['epochs'] - hparams['warmup_epochs'])
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif hparams['scheduler_type'] == 'multistep':
        milestones = [int(hparams['epochs'] * 0.3), int(hparams['epochs'] * 0.6), int(hparams['epochs'] * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    else:
        raise ValueError(f"不支持的调度器类型: {hparams['scheduler_type']}")
    
    return optimizer, scheduler

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = torch.from_numpy(np.random.beta(alpha, alpha, 1)).float().to(x.device)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger=None, rank=0, use_mixup=True):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_mixup and torch.rand(1).item() > 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + 
                       (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 50 == 0 and rank == 0:
            current_acc = 100.0 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            msg = f'Epoch {epoch} 批次 [{batch_idx}/{len(train_loader)}] 损失: {current_loss:.4f} 准确率: {current_acc:.2f}%'
            print(msg)
            if logger:
                logger.info(msg)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    train_time = time.time() - start_time
    
    if rank == 0 and logger:
        logger.info(f'Epoch {epoch} 训练完成 - 时间: {train_time:.2f}s, 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def test_model(model, test_loader, criterion, device, epoch, logger=None, distributed=False, rank=0):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    if distributed:
        test_loss_tensor = torch.tensor(test_loss).to(device)
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        test_loss = test_loss_tensor.item() / dist.get_world_size()
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    test_time = time.time() - start_time
    
    if rank == 0 and logger:
        logger.info(f'Epoch {epoch} 测试完成 - 时间: {test_time:.2f}s, 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%')
    
    return test_loss, test_acc

def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path, logger=None, rank=0):
    """保存检查点"""
    if rank == 0:
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(state, checkpoint_path)
        if logger:
            logger.info(f"检查点已保存: {checkpoint_path}")

if __name__ == '__main__':
    # Test get_hyperparameters
    test_models = [
        'resnet_20', 'eca_resnet_20', 'ghost_resnet_32', 'convnext_tiny', 
        'convnext_tiny_timm', 'coatnet_0', 'mlp_mixer_b16', 'segnext_mscan_tiny'
    ]
    print("--- Testing Hyperparameter Retrieval ---")
    for model_name in test_models:
        try:
            hparams = get_hyperparameters(model_name)
            print(f"Model: {model_name}")
            print(f"  LR: {hparams['lr']}, Optim: {hparams['optimizer_type']}, WD: {hparams['weight_decay']}")
            print(f"  BS/GPU: {hparams['batch_size_per_gpu']}, Scheduler: {hparams['scheduler_type']}")
            print(f"  Warmup: {hparams['warmup_epochs']}, Use ImgNet Norm: {hparams['use_imagenet_norm']}")
            print(f"  Model Params: {hparams['model_constructor_params']}")
            assert hparams['epochs'] == 200
        except ValueError as e:
            print(f"Error for {model_name}: {e}")
    
    # Test save_experiment_results (dummy data)
    print("\n--- Testing Experiment Results Saving with run_label ---")
    dummy_hparams_labeled = get_hyperparameters('eca_resnet_20')
    dummy_hparams_labeled['model_constructor_params'] = {'k_size': 5} # Simulate an override
    save_experiment_results(
        results_data=dummy_results_data, 
        model_name='eca_resnet_20', 
        hparams=dummy_hparams_labeled,
        metrics_history=dummy_metrics_history,
        output_dir='logs/results_test',
        run_label='ECA-ResNet-20 (k_size=5)' # Test with a run label
    )
    assert os.path.exists('logs/results_test/eca_resnet_20_ECA-ResNet-20_k_size-5_results.json')
    print("Dummy results with run_label saved. Check logs/results_test/eca_resnet_20_ECA-ResNet-20_k_size-5_results.json")

    # Clean up test files if desired
    # if os.path.exists('logs/results_test/resnet_20_test_results.json'): os.remove('logs/results_test/resnet_20_test_results.json')
    # if os.path.exists('logs/results_test/eca_resnet_20_ECA-ResNet-20_k_size-5_results.json'): os.remove('logs/results_test/eca_resnet_20_ECA-ResNet-20_k_size-5_results.json')
    # if os.path.exists('logs/results_test') and not os.listdir('logs/results_test'): os.rmdir('logs/results_test')

    print("\nutils.py tests completed.") 