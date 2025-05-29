import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from accelerate import Accelerator, DistributedType
from accelerate.utils import tqdm
import time
import os
import json
from pathlib import Path
import math

from model import get_model
from dataset import get_dataloaders
from utils import (
    get_hyperparameters, 
    count_parameters, 
    plot_training_curves, 
    save_experiment_results,
    REPORT_HYPERPARAMETERS
)

def train_one_epoch(model, train_loader, optimizer, criterion, accelerator, epoch_num, total_epochs):
    model.train()
    total_loss_sum = 0.0 # Sum of losses from all batches on this process
    correct_preds_sum = 0 # Sum of correct predictions from all batches on this process
    total_samples_processed_this_epoch = 0 # Total samples processed across all batches on this process

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Train]", 
                        disable=not accelerator.is_local_main_process, leave=False, dynamic_ncols=True)
    
    for batch_data in progress_bar:
        inputs, targets = batch_data 
        current_batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        accelerator.backward(loss)
        optimizer.step()
        
        # For calculating epoch average loss & acc, gather metrics from all processes
        # Loss from criterion is usually mean over batch, so multiply by local batch size before summing
        # then gather and divide by total global samples.
        batch_loss_sum_across_gpus = accelerator.gather_for_metrics(loss.detach() * current_batch_size).sum()
        
        _, predicted = torch.max(outputs.data, 1)
        batch_correct_preds_across_gpus = accelerator.gather_for_metrics((predicted == targets).sum()).sum()
        batch_total_samples_across_gpus = accelerator.gather_for_metrics(torch.tensor(current_batch_size, device=accelerator.device)).sum()

        # Accumulate for epoch stats (these are now global sums for the batch)
        total_loss_sum += batch_loss_sum_across_gpus.item()
        correct_preds_sum += batch_correct_preds_across_gpus.item()
        total_samples_processed_this_epoch += batch_total_samples_across_gpus.item()
        
        if accelerator.is_local_main_process:
            current_batch_avg_loss = (loss.item()) # Avg loss for this process's batch
            current_batch_acc = (predicted == targets).sum().item() / current_batch_size * 100 if current_batch_size > 0 else 0
            progress_bar.set_postfix(loss=f"{current_batch_avg_loss:.4f}", acc=f"{current_batch_acc:.2f}%")

    # Epoch averages are calculated based on globally summed losses and corrects
    avg_epoch_loss = total_loss_sum / total_samples_processed_this_epoch if total_samples_processed_this_epoch > 0 else 0
    avg_epoch_acc = (correct_preds_sum / total_samples_processed_this_epoch) * 100 if total_samples_processed_this_epoch > 0 else 0
    
    return avg_epoch_loss, avg_epoch_acc

def evaluate(model, test_loader, criterion, accelerator, epoch_num, total_epochs):
    model.eval()
    total_loss_sum = 0.0
    correct_preds_top1_sum = 0
    correct_preds_top5_sum = 0
    total_samples_processed = 0
    
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Test ]", 
                        disable=not accelerator.is_local_main_process, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_data in progress_bar:
            inputs, targets = batch_data
            current_batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Gather metrics from all processes
            batch_loss_sum_across_gpus = accelerator.gather_for_metrics(loss.detach() * current_batch_size).sum()
            
            all_outputs = accelerator.gather_for_metrics(outputs)
            all_targets = accelerator.gather_for_metrics(targets)
            batch_total_samples_across_gpus = all_targets.size(0) # size(0) after gathering is global batch size

            total_loss_sum += batch_loss_sum_across_gpus.item()
            total_samples_processed += batch_total_samples_across_gpus

            # Top-1 accuracy
            _, predicted_top1 = torch.max(all_outputs.data, 1)
            correct_preds_top1_sum += (predicted_top1 == all_targets).sum().item()
            
            # Top-5 accuracy
            _, predicted_top5_indices = torch.topk(all_outputs.data, 5, dim=1)
            target_reshaped = all_targets.view(-1, 1).expand_as(predicted_top5_indices)
            correct_preds_top5_sum += torch.sum(predicted_top5_indices == target_reshaped).item()

            if accelerator.is_local_main_process:
                progress_bar.set_postfix(loss=f"{(loss.item()):.4f}")

    avg_epoch_loss = total_loss_sum / total_samples_processed if total_samples_processed > 0 else 0
    avg_acc_top1 = (correct_preds_top1_sum / total_samples_processed) * 100 if total_samples_processed > 0 else 0
    avg_acc_top5 = (correct_preds_top5_sum / total_samples_processed) * 100 if total_samples_processed > 0 else 0
    return avg_epoch_loss, avg_acc_top1, avg_acc_top5

def run_training_for_model(model_name: str, accelerator: Accelerator, config_override: dict = None):
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print(f"\n\n{'='*80}")
        print(f"🚀 Initializing Training for Model: {model_name} on device: {accelerator.device}")
        print(f"{'='*80}")

    try:
        hparams = get_hyperparameters(model_name)
        if config_override:
            if accelerator.is_local_main_process: print(f"Applying config override: {config_override}")
            hparams.update(config_override)
    except ValueError as e:
        accelerator.print(f"Skipping {model_name}: Hyperparameter retrieval error - {e}")
        return None

    if accelerator.is_local_main_process:
        print(f"Hyperparameters for {model_name}:")
        for key, val in hparams.items():
            if isinstance(val, dict) and key == 'model_constructor_params':
                 print(f"  {key}: {val}")
            else:
                 print(f"  {key}: {val}")

    train_loader, test_loader = get_dataloaders(
        batch_size=hparams['batch_size_per_gpu'],
        use_imagenet_norm=hparams['use_imagenet_norm'],
        num_workers=4 
    )
    if accelerator.is_local_main_process:
        print(f"DataLoaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        print(f"Local batch size: {hparams['batch_size_per_gpu']}. Effective global batch size: {hparams['batch_size_per_gpu'] * accelerator.num_processes}")

    timm_model_keys = ["convnext_tiny_timm", "coatnet_0", "cspresnet50", "ghostnet_100", "hornet_tiny", "resnest50d", "mlp_mixer_b16"]
    should_load_timm_pretrained = model_name in timm_model_keys

    model_constructor_args = hparams.get('model_constructor_params', {})
    model = get_model(
        model_name=model_name, 
        num_classes=100, 
        pretrained_timm=should_load_timm_pretrained, 
        **model_constructor_args
    )
    
    params_m = count_parameters(model)
    if accelerator.is_local_main_process:
        print(f"Model {model_name} created. Parameters: {params_m:.2f}M. Pretrained (timm): {should_load_timm_pretrained}")

    if hparams['optimizer_type'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=hparams['lr'], momentum=0.9, weight_decay=hparams['weight_decay'])
    elif hparams['optimizer_type'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    else:
        accelerator.print(f"Unsupported optimizer_type: {hparams['optimizer_type']}")
        raise ValueError(f"Unsupported optimizer_type: {hparams['optimizer_type']}")

    criterion = nn.CrossEntropyLoss()
    num_epochs = hparams['epochs']
    warmup_epochs = hparams['warmup_epochs']

    if hparams['scheduler_type'] == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6) 
    elif hparams['scheduler_type'] == 'cosine_annealing_warmup':
        def lr_lambda_warmup_cosine(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs + 1e-8) 
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup_cosine)
    else:
        accelerator.print(f"Unsupported scheduler_type: {hparams['scheduler_type']}")
        raise ValueError(f"Unsupported scheduler_type: {hparams['scheduler_type']}")

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    if accelerator.is_local_main_process: print("Accelerator preparation complete.")

    metrics_history = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': [], 'test_accs_top5': []}
    best_test_acc_top1 = 0.0
    start_time_total = time.time()

    if accelerator.is_local_main_process: print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, accelerator, epoch, num_epochs)
        test_loss, test_acc_top1, test_acc_top5 = evaluate(model, test_loader, criterion, accelerator, epoch, num_epochs)
        
        if not accelerator.optimizer_step_was_skipped: 
             scheduler.step()
        
        if accelerator.is_local_main_process:
            metrics_history['train_losses'].append(train_loss)
            metrics_history['train_accs'].append(train_acc)
            metrics_history['test_losses'].append(test_loss)
            metrics_history['test_accs'].append(test_acc_top1)
            metrics_history['test_accs_top5'].append(test_acc_top5)
            epoch_duration = time.time() - epoch_start_time
            accelerator.print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Acc@1: {test_acc_top1:.2f}%, Acc@5: {test_acc_top5:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.6e} | Time: {epoch_duration:.2f}s"
            )
            if test_acc_top1 > best_test_acc_top1:
                best_test_acc_top1 = test_acc_top1
                accelerator.print(f"🎉 New best Acc@1: {best_test_acc_top1:.2f}%. Saving checkpoint... 🎉")
                checkpoint_dir = Path('logs/checkpoints')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                safe_model_name = model_name.replace('/', '_').replace(':', '_')
                save_file_path = checkpoint_dir / f"{safe_model_name}_best.pth"
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), save_file_path)
                accelerator.print(f"Saved best model state_dict to {save_file_path}")
        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        total_training_time_seconds = time.time() - start_time_total
        results_data = {
            'best_test_acc_top1': best_test_acc_top1,
            'final_test_acc_top1': metrics_history['test_accs'][-1] if metrics_history['test_accs'] else 0,
            'final_test_acc_top5': metrics_history['test_accs_top5'][-1] if metrics_history['test_accs_top5'] else 0,
            'final_train_loss': metrics_history['train_losses'][-1] if metrics_history['train_losses'] else float('inf'),
            'final_train_acc': metrics_history['train_accs'][-1] if metrics_history['train_accs'] else 0,
            'params_M': params_m,
            'train_time_total_seconds': total_training_time_seconds,
            'estimated_total_train_hours_8xV100': total_training_time_seconds / 3600 
        }
        plot_training_curves(metrics_history, model_name, save_dir='assets')
        save_experiment_results(
            results_data=results_data, model_name=model_name, hparams=hparams,
            metrics_history=metrics_history, output_dir='logs/results'
        )
        accelerator.print(f"🏁 Finished training {model_name}. Best Acc@1: {best_test_acc_top1:.2f}%. Total time: {total_training_time_seconds/3600:.2f}h")
        accelerator.print(f"Results and plots saved for {model_name}.")
    accelerator.wait_for_everyone()
    return metrics_history

def train_all_models_main(models_to_train_override=None):
    accelerator = Accelerator(mixed_precision=None) 
    if accelerator.is_local_main_process:
        print(f"Process {accelerator.local_process_index} (device: {accelerator.device}) starting..."
              f"Distributed type: {accelerator.distributed_type}, Num processes: {accelerator.num_processes}")
        if str(accelerator.device).startswith('cuda'):
             print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(accelerator.device) if torch.cuda.is_available() else 'N/A'}")
        else:
            print("Using CPU.")

    if models_to_train_override:
        all_model_names = models_to_train_override
        if accelerator.is_local_main_process: print(f"Training a subset of models: {all_model_names}")
    else:
        all_model_names = list(REPORT_HYPERPARAMETERS['model_to_category'].keys())
    
    overall_start_time = time.time()
    completed_models = []
    failed_models = []
    for model_name in all_model_names:
        try:
            run_training_for_model(model_name, accelerator)
            completed_models.append(model_name)
        except Exception as e:
            accelerator.print(f"💥💥 ERROR during training setup or execution for {model_name}: {e} 💥💥")
            import traceback
            accelerator.print(traceback.format_exc())
            failed_models.append(model_name)
        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        overall_duration = time.time() - overall_start_time
        print(f"\n{'='*80}")
        print("🏁 ALL MODEL TRAINING SCRIPT COMPLETE 🏁")
        print(f"Total script duration: {overall_duration/3600:.2f} hours ({overall_duration:.0f} seconds)")
        print(f"Successfully completed model training process for: {len(completed_models)} models")
        if failed_models:
            print(f"Failed/Skipped models: {len(failed_models)}")
            for m in failed_models: print(f"  - {m}")
        print(f"{'='*80}")

if __name__ == "__main__":
    train_all_models_main() 