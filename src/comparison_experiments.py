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
matplotlib.use('Agg') # Ensure Agg backend before pyplot import for headless environments
import matplotlib.pyplot as plt

# Project imports
from model import get_model # Already uses new get_model
from utils import get_hyperparameters, save_experiment_results, plot_training_curves, REPORT_HYPERPARAMETERS, save_comparison_summary_text, compare_models as plot_compare_models
from train_all_models import run_training_for_model # Key import

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff() # Turn off interactive mode for matplotlib

# The ComparisonExperiment class is removed as its functionality is replaced by 
# run_training_for_model and other utilities.

class ArchitectureComparison:
    """Defines groups of models for architecture type comparison."""
    @staticmethod
    def get_model_groups():
        # These groups are illustrative. The actual training will use hyperparameters defined
        # in utils.REPORT_HYPERPARAMETERS for each model.
        # The comparison aspect comes from analyzing their results side-by-side.
        return {
            '基础卷积网络': ['resnet_20', 'resnet_32'],
            '注意力机制': ['eca_resnet_20', 'segnext_mscan_tiny'],
            '轻量化设计': ['ghost_resnet_20', 'ghostnet_100'],
            '现代化架构': ['convnext_tiny', 'convnext_tiny_timm'],
            '混合架构': ['coatnet_0', 'resnest50d'],
            'MLP架构': ['mlp_mixer_tiny', 'mlp_mixer_b16']
        }

class EfficiencyComparison:
    """Defines models for efficiency (Params vs Accuracy) comparison."""
    @staticmethod
    def get_efficiency_models():
        return [
            'resnet_20', 'eca_resnet_20', 'ghost_resnet_20',
            'convnext_tiny', 'ghostnet_100', 'segnext_mscan_tiny', 'mlp_mixer_tiny'
        ]

class PretrainedVsFromScratch:
    """Defines model pairs for pretrained vs. from-scratch comparison."""
    @staticmethod
    def get_comparison_pairs():
        # Each tuple: (from_scratch_variant_name, timm_pretrained_variant_name)
        # The training script will need to handle the `pretrained_timm` flag appropriately.
        return [
            # For convnext_tiny, we have a custom 'convnext_tiny' and 'convnext_tiny_timm'
            # 'convnext_tiny' will be trained from scratch (should_load_timm_pretrained=False in run_training_for_model)
            # 'convnext_tiny_timm' will use pretrained (should_load_timm_pretrained=True)
            {'name': 'convnext_tiny', 'pretrained_flag_override': False, 'label': 'ConvNeXt-T (Scratch)'},
            {'name': 'convnext_tiny_timm', 'pretrained_flag_override': True, 'label': 'ConvNeXt-T (Pretrained)'},
            
            # For mlp_mixer_tiny vs mlp_mixer_b16
            # 'mlp_mixer_tiny' is custom from scratch
            # 'mlp_mixer_b16' is timm and typically pretrained
            {'name': 'mlp_mixer_tiny', 'pretrained_flag_override': False, 'label': 'MLP-Mixer-T (Scratch)'},
            {'name': 'mlp_mixer_b16', 'pretrained_flag_override': True, 'label': 'MLP-Mixer-B/16 (Pretrained)'}
        ]

def run_selected_experiments(experiment_configs, accelerator, experiment_name="Selected Experiments"):
    """
    Runs a list of specified model training configurations.
    Args:
        experiment_configs (list of dict): Each dict contains 'model_name' and optional 'config_override'.
        accelerator (Accelerator): The Hugging Face Accelerator instance.
        experiment_name (str): Name for logging.
    """
    if accelerator.is_local_main_process:
        print(f"\n\n{'='*80}")
        print(f"🔬 Starting {experiment_name} 🔬")
        print(f"{'='*80}")

    all_results_summary = {} # To store paths or key metrics for final summary plot

    for config in experiment_configs:
        model_name = config['model_name']
        config_override = config.get('config_override', {})
        # The `label` can be used for custom naming in plots if needed later
        # label = config.get('label', model_name)

        if accelerator.is_local_main_process:
            print(f"--- Running for: {model_name} (Override: {config_override}) ---")
        
        # run_training_for_model now saves its own detailed JSON output
        # It returns metrics_history, which might not be needed here directly if we read from JSONs later
        run_training_for_model(model_name, accelerator, config_override=config_override)
        
        # For a summary plot at the end of these comparison runs, we'd need to collect paths
        # to the generated JSON files or the key metrics themselves.
        # This part can be enhanced if Comparison-specific plots are made here.
        # For now, individual JSONs are generated by run_training_for_model.
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        if config_override:
             # If overrides create a unique variant, reflect in name for results lookup
             override_suffix = "_" + "_".join([f"{k}_{v}" for k,v in config_override.items()])
             override_suffix = override_suffix.replace('.', 'p').replace('-','neg') # Sanitize further
             safe_model_name += override_suffix
        # This assumes result file naming convention used in run_training_for_model
        # This is not strictly true as run_training_for_model uses the original model_name for saving. Careful here.
        # For now, we assume generate_results.py will pick them up by original model name.

    if accelerator.is_local_main_process:
        print(f"\n---- {experiment_name} Complete ----")
        print("Individual model training results and plots saved in logs/results and assets/ respectively.")
        print("Run generate_results.py to create summary tables and comparison plots from these JSON files.")

def run_architecture_comparison_main(accelerator):
    if accelerator.is_local_main_process: print("\n--- Running Architecture Comparison Experiments ---")
    groups = ArchitectureComparison.get_model_groups()
    experiment_configs = []
    for category, models_in_group in groups.items():
        if accelerator.is_local_main_process: print(f"Category: {category}")
        for model_name in models_in_group:
            if model_name in REPORT_HYPERPARAMETERS['model_to_category']: # Check if model is known
                experiment_configs.append({'model_name': model_name})
            else:
                if accelerator.is_local_main_process: print(f"Warning: Model {model_name} from architecture groups not in main hyperparameter list. Skipping.")
    run_selected_experiments(experiment_configs, accelerator, "Architecture Comparison")

def run_efficiency_comparison_main(accelerator):
    if accelerator.is_local_main_process: print("\n--- Running Efficiency Comparison Experiments ---")
    models_for_efficiency = EfficiencyComparison.get_efficiency_models()
    experiment_configs = []
    for model_name in models_for_efficiency:
        if model_name in REPORT_HYPERPARAMETERS['model_to_category']:
            experiment_configs.append({'model_name': model_name})
        else:
             if accelerator.is_local_main_process: print(f"Warning: Model {model_name} from efficiency list not in main hyperparameter list. Skipping.")
    run_selected_experiments(experiment_configs, accelerator, "Efficiency Comparison")

def run_pretrained_vs_scratch_main(accelerator):
    if accelerator.is_local_main_process: print("\n--- Running Pretrained vs. From Scratch Comparison ---")
    pairs = PretrainedVsFromScratch.get_comparison_pairs()
    experiment_configs = []
    for pair_config in pairs:
        model_name = pair_config['name']
        # The `pretrained_timm` flag is handled by `get_model` based on model_name convention or explicit call.
        # `run_training_for_model` decides `should_load_timm_pretrained` based on `model_name`.
        # If we need to force from-scratch for a model that `get_model` might make pretrained by default,
        # or force pretraining for one it might not, we'd need a more direct override.
        # For now, the `should_load_timm_pretrained` in `run_training_for_model` uses a heuristic.
        # We can add an explicit `load_pretrained` to `config_override` if needed.
        
        # Let's assume get_hyperparameters gives the base, and we might want to override `use_imagenet_norm` too.
        hparams_base = get_hyperparameters(model_name)
        config_override = {}

        label = pair_config['label'] # e.g. "ConvNeXt-T (Scratch)" or "ConvNeXt-T (Pretrained)"

        # How to ensure one runs pretrained and other from scratch?
        # `run_training_for_model`'s `should_load_timm_pretrained` logic:
        # `model_name in timm_model_keys` (e.g. "convnext_tiny_timm" is True, "convnext_tiny" is False)
        # This setup from `get_comparison_pairs` already distinguishes the models.
        # We don't need a specific override for pretraining flag if model names are distinct and handled by `run_training_for_model`.

        if "(Scratch)" in label:
            # For a scratch version, ensure imagenet norm is False if it defaults to True for this model category
            if hparams_base.get('use_imagenet_norm', False):
                config_override['use_imagenet_norm'] = False 
        elif "(Pretrained)" in label:
            # For a pretrained version, ensure imagenet norm is True
            config_override['use_imagenet_norm'] = True

        experiment_configs.append({
            'model_name': model_name, 
            'config_override': config_override,
            'label': label
        })        
    run_selected_experiments(experiment_configs, accelerator, "Pretrained vs. From Scratch")

# Plotting functions like create_comparison_plots are removed for now.
# Such plots should be generated by reading the JSON results, possibly in generate_results.py
# or a dedicated analysis script.

def main():
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Starting Comparison Experiments Orchestration Script")
        Path("logs/results").mkdir(parents=True, exist_ok=True)
        Path("logs/checkpoints").mkdir(parents=True, exist_ok=True)
        Path("assets").mkdir(parents=True, exist_ok=True)

    # Select which comparisons to run
    run_architecture_comparison_main(accelerator)
    accelerator.wait_for_everyone()

    run_efficiency_comparison_main(accelerator)
    accelerator.wait_for_everyone()
    
    run_pretrained_vs_scratch_main(accelerator)
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        print("\nComparison experiments orchestration finished.")
        print("Individual results are in logs/results/. Run generate_results.py for summaries.")

if __name__ == "__main__":
    main() 