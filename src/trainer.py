import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import time
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, fields
import json

from .utils import (
    setup_logging, log_system_info, setup_distributed, cleanup_distributed,
    get_optimizer_scheduler, save_checkpoint, plot_training_curves,
    save_experiment_results, mixup_data, mixup_criterion, get_hyperparameters
)

@dataclass
class TrainingArguments:
    """训练参数配置类，参考transformers.TrainingArguments"""
    
    output_dir: str = "./logs"  # 基础输出目录, e.g., from train.py
    overwrite_output_dir: bool = False # Not currently used, but good to have
    
    num_train_epochs: int = 200
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 256
    
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    warmup_epochs: int = 0
    lr_scheduler_type: str = "cosine_annealing"
    
    logging_steps: int = 50
    eval_strategy: str = "epoch" # Evaluate at the end of each epoch
    save_strategy: str = "epoch" # Save at the end of each epoch if it's best or matches save_steps
    save_steps: int = 20 # Save a checkpoint every N epochs (used if save_strategy is 'steps', or for periodic saves)
    
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    
    label_smoothing_factor: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    
    optimizer_type: str = "sgd"
    use_imagenet_norm: bool = False
    model_constructor_params: dict = field(default_factory=dict)
    
    resume_from_checkpoint: Optional[str] = None

    model_name_for_log: Optional[str] = None # Set by from_model_name
    run_name: Optional[str] = None # User-specified run name, passed from train.py via kwargs
    
    # Default filenames (these will be saved within the final effective_output_dir)
    checkpoint_filename_best: str = "best_model.pth"
    checkpoint_filename_epoch: str = "checkpoint_epoch_{epoch}.pth" # Placeholder for epoch number
    evaluation_filename: str = "evaluation_summary.json"
    # Combined plot for loss and accuracy is typical, use loss_plot_filename for it
    plot_filename: str = "training_curves.png"
    
    def __post_init__(self):
        # This method is now simplified. The base output_dir (e.g., ./logs)
        # existence will be checked/created by the Trainer's main process.
        # The full, run-specific path is constructed and managed within the Trainer.
        pass
    
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs):
        hparams = get_hyperparameters(model_name)
        current_config = hparams.copy()
        current_config.update(kwargs) # kwargs from train.py (like output_dir, run_name, epochs, lr) override hparams
        
        current_config['model_name_for_log'] = model_name # Store the original model_name

        # Remap lr and batch_size keys if they come from hparams
        if 'learning_rate' not in current_config and 'lr' in current_config:
            current_config['learning_rate'] = current_config.pop('lr')
        
        if 'per_device_train_batch_size' not in current_config and 'batch_size_per_gpu' in current_config:
            current_config['per_device_train_batch_size'] = current_config.pop('batch_size_per_gpu')
            # Assume eval batch size is related if not explicitly set in kwargs
            if 'per_device_eval_batch_size' not in current_config:
                 current_config['per_device_eval_batch_size'] = current_config['per_device_train_batch_size'] * 2

        valid_field_names = {f.name for f in fields(cls)}
        init_args = {k: v for k, v in current_config.items() if k in valid_field_names}
        return cls(**init_args)

class Trainer:
    """通用训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_metrics: Optional[callable] = None, # Not used yet, but placeholder
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self.compute_metrics = compute_metrics
        
        self.distributed, self.rank, self.world_size, self.gpu = setup_distributed()
        
        # 如果没有设置分布式环境但有多个GPU可用，自动使用DataParallel
        self.use_data_parallel = False
        if not self.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.use_data_parallel = True
            self.world_size = torch.cuda.device_count()
            print(f'检测到 {torch.cuda.device_count()} 个GPU，将使用DataParallel进行多GPU训练')
        
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.distributed else 'cuda' if torch.cuda.is_available() else 'cpu')

        # 构建 effective_output_dir
        # effective_run_name: user-provided run_name or a timestamp
        self.effective_run_name = self.args.run_name if self.args.run_name else time.strftime("%Y%m%d-%H%M%S")
        model_log_name = self.args.model_name_for_log if self.args.model_name_for_log else "unknown_model"
        
        self.effective_output_dir = os.path.join(
            self.args.output_dir, # Base dir, e.g., ./logs
            model_log_name,      # Model-specific subdir, e.g., resnet_50
            self.effective_run_name # Run-specific subdir, e.g., my_experiment_run or 20231027-153000
        )

        if self.rank == 0:
            os.makedirs(self.args.output_dir, exist_ok=True) # Ensure base ./logs exists
            os.makedirs(self.effective_output_dir, exist_ok=True) # Create the full run-specific path
            
        self.logger = setup_logging(self.effective_output_dir, self.rank) # Logger uses the full path
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"所有输出将保存到: {self.effective_output_dir}")
            log_system_info(self.logger, self.rank)
            self.logger.info(f'TrainingArguments: {json.dumps(self.args.__dict__, indent=2)}')
            if self.distributed:
                self.logger.info(f'分布式训练 (DDP): True, Rank: {self.rank}, World Size: {self.world_size}')
            elif self.use_data_parallel:
                self.logger.info(f'多GPU训练 (DataParallel): True, GPU数量: {self.world_size}')
            else:
                self.logger.info(f'单GPU训练: True')
            self.logger.info(f'使用设备: {self.device}')
        
        self.model = self.model.to(self.device)
        if self.distributed:
            # find_unused_parameters can be True if model has parts not used in forward pass during DDP
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=False) 
        elif self.use_data_parallel:
            # 使用DataParallel进行多GPU训练
            self.model = nn.DataParallel(self.model)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_factor)
        self.best_metric = 0.0 # Typically, higher is better (e.g., accuracy)
        self.global_step = 0
        self.epoch = 0
        self.train_losses, self.train_accs, self.eval_losses, self.eval_accs = [], [], [], []
    
    def get_train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True) if self.distributed else None
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.per_device_train_batch_size, 
            sampler=sampler,
            shuffle=(sampler is None), 
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory, 
            drop_last=self.args.dataloader_drop_last
        )
    
    def get_eval_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.eval_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False) if self.distributed else None
        return DataLoader(
            self.eval_dataset, 
            batch_size=self.args.per_device_eval_batch_size, 
            sampler=sampler,
            shuffle=False, 
            num_workers=self.args.dataloader_num_workers, 
            pin_memory=self.args.dataloader_pin_memory
        )
    
    def create_optimizer_and_scheduler(self, num_training_steps_per_epoch: int):
        lr = self.args.learning_rate
        
        if self._optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
            ]
            if self.args.optimizer_type.lower() == "sgd": 
                self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9, nesterov=True)
            elif self.args.optimizer_type.lower() == "adamw": 
                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=self.args.adam_epsilon)
            else: 
                raise ValueError(f"不支持的优化器类型: {self.args.optimizer_type}")
        else: 
            self.optimizer = self._optimizer
        
        total_epochs_for_scheduler = self.args.num_train_epochs
        if self._lr_scheduler is None:
            if self.args.lr_scheduler_type.lower() in ["cosine_annealing", "cosine"]:
                 self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs_for_scheduler, eta_min=1e-6)
            elif self.args.lr_scheduler_type.lower() in ["cosine_annealing_warmup", "cosine_warmup"]:
                if self.args.warmup_epochs > 0:
                    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.args.warmup_epochs)
                    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(total_epochs_for_scheduler - self.args.warmup_epochs), eta_min=1e-6)
                    self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler_warmup, scheduler_main], milestones=[self.args.warmup_epochs])
                else:
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs_for_scheduler, eta_min=1e-6)
            elif self.args.lr_scheduler_type.lower() == "multistep": 
                milestones = [int(total_epochs_for_scheduler * 0.5), int(total_epochs_for_scheduler * 0.75)]
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            else: 
                raise ValueError(f"不支持的调度器类型: {self.args.lr_scheduler_type}")
        else: 
            self.lr_scheduler = self._lr_scheduler

    def train(self):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        steps_per_epoch = len(train_dataloader)
        self.create_optimizer_and_scheduler(steps_per_epoch)
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"***** 开始训练 *****")
            self.logger.info(f"  训练模型: {self.args.model_name_for_log}, 运行: {self.effective_run_name}")
            self.logger.info(f"  训练样本数 = {len(self.train_dataset)}, 每轮步骤数 = {steps_per_epoch}")
            self.logger.info(f"  总轮数 = {self.args.num_train_epochs}")
            self.logger.info(f"  每设备批次大小 = {self.args.per_device_train_batch_size}")
            if self.distributed or self.use_data_parallel:
                self.logger.info(f"  总批次大小 (所有GPU) = {self.args.per_device_train_batch_size * self.world_size}")
            else:
                self.logger.info(f"  单GPU批次大小 = {self.args.per_device_train_batch_size}")
            self.logger.info(f"  初始学习率 = {self.optimizer.param_groups[0]['lr']:.2e}")
        
        start_time = time.time()
        for epoch_idx in range(self.args.num_train_epochs):
            self.epoch = epoch_idx # self.epoch is 0-indexed
            if self.distributed and hasattr(train_dataloader, 'sampler'): train_dataloader.sampler.set_epoch(self.epoch)
            
            train_loss, train_acc = self._train_epoch(train_dataloader, self.epoch)
            eval_loss, eval_acc = self._eval_epoch(eval_dataloader, self.epoch) # eval_acc is top-1
            
            # Scheduler step should happen after optimizer.step() in the epoch, typically once per epoch for epoch-based schedulers
            self.lr_scheduler.step() 
            
            if self.rank == 0:
                self.train_losses.append(train_loss); self.train_accs.append(train_acc)
                self.eval_losses.append(eval_loss); self.eval_accs.append(eval_acc)
                
                is_best = eval_acc > self.best_metric
                if is_best: 
                    self.best_metric = eval_acc
                    self._save_checkpoint(filename=self.args.checkpoint_filename_best, is_best=True)
                    if self.logger: self.logger.info(f'Epoch {self.epoch+1}: 新的最佳准确率: {self.best_metric:.2f}% (已保存至 {self.args.checkpoint_filename_best})')
                
                # Periodic checkpoint saving based on save_steps (interpreted as epoch interval here)
                if self.args.save_steps > 0 and (self.epoch + 1) % self.args.save_steps == 0:
                    epoch_checkpoint_filename = self.args.checkpoint_filename_epoch.format(epoch=self.epoch+1)
                    self._save_checkpoint(filename=epoch_checkpoint_filename)
        
        if self.rank == 0: self._log_final_results(time.time() - start_time)
        if self.distributed: cleanup_distributed()
    
    def _train_epoch(self, dataloader: DataLoader, current_epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            
            if self.args.use_mixup and self.model.training and torch.rand(1).item() > 0.5: # Apply mixup randomly
                mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=self.args.mixup_alpha)
                outputs = self.model(mixed_images)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                with torch.no_grad(): preds = outputs.argmax(dim=-1)
                total_correct += (lam * preds.eq(targets_a).sum().float() + (1.0 - lam) * preds.eq(targets_b).sum().float()).item()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                with torch.no_grad(): preds = outputs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0) # Accumulate loss scaled by batch size
            total_samples += images.size(0)
            self.global_step += 1
            
            if step % self.args.logging_steps == 0 and self.rank == 0 and self.logger:
                current_lr = self.optimizer.param_groups[0]['lr']
                # 对于DataParallel，实际处理的样本数需要考虑多GPU
                effective_world_size = self.world_size if (self.distributed or self.use_data_parallel) else 1
                samples_processed = step * self.args.per_device_train_batch_size * effective_world_size
                batches_per_epoch = len(dataloader)
                log_loss = total_loss / total_samples if total_samples > 0 else 0 # Avg loss so far in epoch
                log_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
                self.logger.info(f'Epoch {current_epoch+1}/{self.args.num_train_epochs} | Batch {step}/{batches_per_epoch} | Loss: {log_loss:.4f} | Acc: {log_acc:.2f}% | LR: {current_lr:.2e}')
        
        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_epoch_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        if self.rank == 0 and self.logger: 
            self.logger.info(f'Epoch {current_epoch+1} TRAIN Summary: Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.2f}%, Duration: {epoch_duration:.2f}s')
        return avg_epoch_loss, avg_epoch_acc
    
    def _eval_epoch(self, dataloader: DataLoader, current_epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        epoch_start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0) # Accumulate loss scaled by batch size
                preds = outputs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item(); total_samples += images.size(0)
        
        # For DDP: gather results from all processes to rank 0
        if self.distributed:
            # Sum up totals from all GPUs
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_correct_tensor = torch.tensor(total_correct, device=self.device)
            total_samples_tensor = torch.tensor(total_samples, device=self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            
            # Convert back to Python numbers on rank 0
            total_loss = total_loss_tensor.item() # This is sum of losses from all batches on all GPUs
            total_correct = total_correct_tensor.item()
            total_samples = total_samples_tensor.item()
        
        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_epoch_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        epoch_duration = time.time() - epoch_start_time

        if self.rank == 0 and self.logger: 
            self.logger.info(f'Epoch {current_epoch+1} EVAL  Summary: Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.2f}%, Duration: {epoch_duration:.2f}s')
        return avg_epoch_loss, avg_epoch_acc
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Saves checkpoint to self.effective_output_dir with the given filename."""
        if self.rank == 0: # Only main process saves checkpoints
            checkpoint_path = os.path.join(self.effective_output_dir, filename)
            # Ensure model is on CPU before saving to avoid GPU-specific tensor storage issues (optional, but good practice)
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            # model_state_dict = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
            model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

            state = {
                'epoch': self.epoch + 1, # Save as 1-indexed
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'best_metric': self.best_metric,
                'global_step': self.global_step,
                'training_args': self.args.__dict__, # Save TrainingArguments as dict for reference
            }
            torch.save(state, checkpoint_path)
            if self.logger:
                checkpoint_type = "最佳模型" if is_best else "检查点"
                self.logger.info(f"Epoch {self.epoch+1}: {checkpoint_type}已保存至 {checkpoint_path}")
    
    def _log_final_results(self, training_time_seconds: float):
        """Logs final training results and saves metrics/plots to self.effective_output_dir."""
        if self.rank == 0: # Ensure only main process performs these actions
            training_time_hours = training_time_seconds / 3600
            msg = f'训练完成! 总用时: {training_time_hours:.2f} 小时。最佳测试准确率: {self.best_metric:.2f}%'
            print(msg) # Keep a simple print for quick console feedback
            if self.logger: self.logger.info(msg)
            
            metrics_history = {
                'train_losses': self.train_losses, 'train_accs': self.train_accs,
                'test_losses': self.eval_losses,   'test_accs': self.eval_accs # Assuming eval_accs stores top-1
            }
            
            # Get model parameters after training (from the potentially DDP-wrapped model on rank 0)
            final_model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            params_m = sum(p.numel() for p in final_model_for_params.parameters() if p.requires_grad) / 1e6

            results_data = {
                'model_name': self.args.model_name_for_log,
                'run_name': self.effective_run_name,
                'best_test_accuracy_top1': self.best_metric,
                'final_test_accuracy_top1': self.eval_accs[-1] if self.eval_accs else 0,
                'training_time_hours': training_time_hours,
                'total_epochs_trained': self.epoch + 1, # self.epoch is 0-indexed
                'global_steps_completed': self.global_step,
                'parameters_M': params_m
            }
            
            # Save detailed experiment results (JSON)
            save_experiment_results(
                results_data=results_data,
                model_name=self.args.model_name_for_log if self.args.model_name_for_log else "unknown_model",
                hparams=self.args.__dict__, # Save the TrainingArguments dict
                output_dir=self.effective_output_dir, # Use the full run-specific path
                metrics_history=metrics_history,
                run_label=self.effective_run_name, # Use the generated or provided run name
                filename=self.args.evaluation_filename # Use filename from TrainingArguments
            )
            
            # Plot training curves
            plot_training_curves(
                metrics_history, 
                title_prefix=f"{self.args.model_name_for_log} ({self.effective_run_name})",
                output_dir=self.effective_output_dir, # Use the full run-specific path
                # Assuming plot_training_curves saves a single combined image using one of these names
                # or handles them internally. The new default is 'plot_filename'.
                loss_filename=self.args.plot_filename, # Pass the general plot filename
                accuracy_filename=self.args.plot_filename # Pass the same if it's a combined plot
            ) 