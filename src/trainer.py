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

from utils import (
    setup_logging, log_system_info, setup_distributed, cleanup_distributed,
    get_optimizer_scheduler, save_checkpoint, plot_training_curves,
    save_experiment_results, mixup_data, mixup_criterion, get_hyperparameters
)

@dataclass
class TrainingArguments:
    """训练参数配置类，参考transformers.TrainingArguments"""
    
    output_dir: str = "./logs"
    overwrite_output_dir: bool = False
    
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
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_steps: int = 20
    
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
    
    def __post_init__(self):
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs):
        hparams = get_hyperparameters(model_name) # Contains 'lr', 'batch_size_per_gpu', etc.
        
        current_config = hparams.copy()
        # kwargs might include 'epochs', 'batch_size_per_device', 'learning_rate' (if --lr was passed),
        # 'output_dir', 'run_name', etc.
        current_config.update(kwargs)

        # Rename keys from hparams/utils format to TrainingArguments field names
        # BEFORE filtering for valid fields. Allow direct field names in kwargs to take precedence.

        # Handle learning_rate mapping:
        # If 'learning_rate' is already in current_config (e.g., from --lr in train.py),
        # it takes precedence. We can remove 'lr' if it also exists to avoid confusion.
        if 'learning_rate' in current_config:
            current_config.pop('lr', None) # Remove 'lr' if 'learning_rate' is already set
        elif 'lr' in current_config: # 'learning_rate' not set, but 'lr' from hparams is available
            current_config['learning_rate'] = current_config.pop('lr')
        
        # Handle batch_size mapping:
        # If 'batch_size_per_device' is in current_config (e.g. from --batch_size in train.py),
        # it takes precedence. Remove 'batch_size_per_gpu' if it also exists.
        if 'batch_size_per_device' in current_config:
            current_config.pop('batch_size_per_gpu', None)
        elif 'batch_size_per_gpu' in current_config: # 'batch_size_per_device' not set, but 'batch_size_per_gpu' from hparams is available
            current_config['batch_size_per_device'] = current_config.pop('batch_size_per_gpu')

        valid_fields = {f.name for f in fields(cls)}
        init_args = {k: v for k, v in current_config.items() if k in valid_fields}
        
        # For debugging, print what's being used:
        # print(f"DEBUG: Initializing TrainingArguments with: {init_args}")

        return cls(**init_args)

class Trainer:
    """训练器类，参考transformers.Trainer设计"""
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_metrics: Optional[callable] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self.compute_metrics = compute_metrics
        
        self.distributed, self.rank, self.world_size, self.gpu = setup_distributed()
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        
        self.logger = setup_logging(args.output_dir, self.rank)
        
        if self.rank == 0:
            log_system_info(self.logger, self.rank)
            if self.logger:
                self.logger.info(f'分布式训练: {self.distributed}')
                self.logger.info(f'世界大小: {self.world_size}')
                self.logger.info(f'使用设备: {self.device}')
        
        self.model = self.model.to(self.device)
        
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=False)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_factor)
        
        self.best_metric = 0.0
        self.global_step = 0
        self.epoch = 0
        self.train_losses = []
        self.train_accs = []
        self.eval_losses = []
        self.eval_accs = []
    
    def get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        if self.distributed:
            sampler = DistributedSampler(
                self.train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank
            )
        else:
            sampler = None
        
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
        """获取评估数据加载器"""
        if self.distributed:
            sampler = DistributedSampler(
                self.eval_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
        else:
            sampler = None
        
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """创建优化器和调度器"""
        if self._optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            lr = self.args.learning_rate * self.world_size if self.distributed else self.args.learning_rate
            
            if self.args.optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD(
                    optimizer_grouped_parameters,
                    lr=lr,
                    momentum=0.9,
                    nesterov=True
                )
            elif self.args.optimizer_type == "adamw":
                self.optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    eps=self.args.adam_epsilon
                )
            else:
                raise ValueError(f"不支持的优化器类型: {self.args.optimizer_type}")
        else:
            self.optimizer = self._optimizer
        
        if self._lr_scheduler is None:
            if self.args.lr_scheduler_type in ["cosine_annealing", "cosine"]:
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.args.num_train_epochs,
                    eta_min=1e-6
                )
            elif self.args.lr_scheduler_type in ["cosine_annealing_warmup", "cosine_warmup"]:
                def lr_lambda(epoch):
                    if epoch < self.args.warmup_epochs:
                        return epoch / self.args.warmup_epochs
                    else:
                        progress = (epoch - self.args.warmup_epochs) / (self.args.num_train_epochs - self.args.warmup_epochs)
                        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            elif self.args.lr_scheduler_type == "multistep":
                milestones = [int(self.args.num_train_epochs * 0.3), 
                             int(self.args.num_train_epochs * 0.6), 
                             int(self.args.num_train_epochs * 0.8)]
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=milestones, gamma=0.2
                )
            else:
                raise ValueError(f"不支持的调度器类型: {self.args.lr_scheduler_type}")
        else:
            self.lr_scheduler = self._lr_scheduler
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步训练"""
        model.train()
        
        input_ids = inputs["input_ids"].to(self.device)
        labels = inputs["labels"].to(self.device)
        
        if self.args.use_mixup and torch.rand(1).item() > 0.5:
            input_ids, labels_a, labels_b, lam = mixup_data(input_ids, labels, alpha=self.args.mixup_alpha)
            outputs = model(input_ids)
            loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(input_ids)
            loss = self.criterion(outputs, labels)
        
        return loss
    
    def evaluation_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """单步评估"""
        model.eval()
        
        input_ids = inputs["input_ids"].to(self.device)
        labels = inputs["labels"].to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            loss = self.criterion(outputs, labels)
            
            preds = outputs.argmax(dim=-1)
            correct = (preds == labels).sum()
            total = labels.size(0)
        
        return {
            "loss": loss,
            "correct": correct,
            "total": total
        }
    
    def train(self):
        """主训练循环"""
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        
        num_training_steps = len(train_dataloader) * self.args.num_train_epochs
        self.create_optimizer_and_scheduler(num_training_steps)
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"***** 开始训练 *****")
            self.logger.info(f"  训练样本数 = {len(self.train_dataset)}")
            self.logger.info(f"  轮数 = {self.args.num_train_epochs}")
            self.logger.info(f"  每设备批次大小 = {self.args.per_device_train_batch_size}")
            self.logger.info(f"  总批次大小 = {self.args.per_device_train_batch_size * self.world_size}")
            self.logger.info(f"  学习率 = {self.optimizer.param_groups[0]['lr']}")
        
        start_time = time.time()
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            
            if self.distributed and hasattr(train_dataloader, 'sampler'):
                train_dataloader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = self._train_epoch(train_dataloader, epoch)
            eval_loss, eval_acc = self._eval_epoch(eval_dataloader, epoch)
            
            self.lr_scheduler.step()
            
            if self.rank == 0:
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.eval_losses.append(eval_loss)
                self.eval_accs.append(eval_acc)
                
                is_best = eval_acc > self.best_metric
                if is_best:
                    self.best_metric = eval_acc
                    self._save_checkpoint(f"{self.args.output_dir}/best_model.pth", is_best=True)
                    if self.logger:
                        self.logger.info(f'新的最佳准确率: {self.best_metric:.2f}%')
                
                if (epoch + 1) % self.args.save_steps == 0:
                    self._save_checkpoint(f"{self.args.output_dir}/checkpoint_epoch_{epoch+1}.pth")
        
        if self.rank == 0:
            training_time = time.time() - start_time
            self._log_final_results(training_time)
        
        cleanup_distributed()
    
    def _train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for step, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.args.use_mixup and self.model.training and torch.rand(1).item() > 0.5:
                mixed_images, targets_a, targets_b, lam = mixup_data(
                    images,
                    labels,
                    alpha=self.args.mixup_alpha
                )
                outputs = self.model(mixed_images)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                total_correct += (lam * preds.eq(targets_a).sum().float() + \
                                 (1 - lam) * preds.eq(targets_b).sum().float()).item()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            total_samples += labels.size(0)
            
            self.global_step += 1
            
            if step % self.args.logging_steps == 0 and self.rank == 0:
                current_acc = 100.0 * total_correct / total_samples
                current_loss = total_loss / (step + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                msg = f'Epoch {epoch+1} 步骤 [{step}/{len(dataloader)}] 损失: {current_loss:.4f} 准确率: {current_acc:.2f}% 学习率: {current_lr:.6f}'
                print(msg)
                if self.logger:
                    self.logger.info(msg)
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100.0 * total_correct / total_samples
        
        if self.rank == 0 and self.logger:
            self.logger.info(f'Epoch {epoch+1} 训练完成 - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def _eval_epoch(self, dataloader, epoch):
        """评估一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0) # loss.item() is avg loss for batch
                preds = outputs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        if self.distributed:
            loss_tensor = torch.tensor(total_loss).to(self.device)
            correct_tensor = torch.tensor(total_correct).to(self.device)
            samples_tensor = torch.tensor(total_samples).to(self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = loss_tensor.item() / self.world_size
            total_correct = correct_tensor.item()
            total_samples = samples_tensor.item()
        
        epoch_loss = total_loss / total_samples if total_samples > 0 else 0 # Use total_loss directly if not averaging per batch before
        epoch_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        
        if self.rank == 0 and self.logger:
            self.logger.info(f'Epoch {epoch+1} 评估完成 - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self, checkpoint_path, is_best=False):
        """保存检查点"""
        state = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'args': self.args,
        }
        
        torch.save(state, checkpoint_path)
        if self.logger:
            checkpoint_type = "最佳模型" if is_best else "检查点"
            self.logger.info(f"{checkpoint_type}已保存: {checkpoint_path}")
    
    def _log_final_results(self, training_time):
        """记录最终结果"""
        msg = f'训练完成! 用时: {training_time/3600:.2f} 小时，最佳准确率: {self.best_metric:.2f}%'
        print(msg)
        if self.logger:
            self.logger.info(msg)
        
        metrics_history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.eval_losses,
            'test_accs': self.eval_accs
        }
        
        results_data = {
            'best_accuracy': self.best_metric,
            'final_accuracy': self.eval_accs[-1] if self.eval_accs else 0,
            'training_time_hours': training_time / 3600,
            'total_epochs': self.args.num_train_epochs,
            'global_steps': self.global_step,
            'parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        save_experiment_results(
            results_data=results_data,
            model_name="resnet_56",
            hparams=self.args.__dict__,
            output_dir=self.args.output_dir,
            metrics_history=metrics_history,
            run_label='trainer_improved'
        )
        
        plot_training_curves(
            metrics_history, 
            "ResNet-56", 
            self.args.output_dir, 
            'resnet56_trainer'
        ) 