#!/usr/bin/env python3
import sys
import argparse
import os
import torch
import json

from .model import get_model, get_model_info
from .dataset import get_cifar100_datasets
from .trainer import Trainer, TrainingArguments
from .utils import get_num_classes # 假设utils.py中有这个函数

def run_training_config(model_name: str, cli_args_dict: dict = None, programmatic_config_override: dict = None):
    """
    为特定模型配置运行训练。
    来自 CLI 和编程调用的覆盖参数将被合并。
    programmatic_config_override 中的 'run_name' 和 'model_constructor_params' 会被特别处理。
    其他 programmatic_config_override 中的键如果匹配 TrainingArguments 的字段，也会作为构造参数。
    """
    train_args_constructor_kwargs = {}

    # 优先处理来自 programmatic_config_override 的 run_name
    if programmatic_config_override and 'run_name' in programmatic_config_override:
        train_args_constructor_kwargs['run_name'] = programmatic_config_override['run_name']
    elif cli_args_dict and 'run_name' in cli_args_dict and cli_args_dict['run_name'] is not None:
        train_args_constructor_kwargs['run_name'] = cli_args_dict['run_name']
    
    # 处理来自 cli_args_dict 的 TrainingArguments 字段覆盖
    if cli_args_dict:
        if 'epochs' in cli_args_dict and cli_args_dict['epochs'] is not None:
            train_args_constructor_kwargs['num_train_epochs'] = cli_args_dict['epochs']
        if 'batch_size' in cli_args_dict and cli_args_dict['batch_size'] is not None:
            train_args_constructor_kwargs['per_device_train_batch_size'] = cli_args_dict['batch_size']
            train_args_constructor_kwargs['per_device_eval_batch_size'] = cli_args_dict['batch_size'] * 2 # 假设评估批次是训练的两倍
        if 'lr' in cli_args_dict and cli_args_dict['lr'] is not None:
            train_args_constructor_kwargs['learning_rate'] = cli_args_dict['lr']
        if 'output_dir' in cli_args_dict and cli_args_dict['output_dir'] is not None:
             train_args_constructor_kwargs['output_dir'] = cli_args_dict['output_dir']

    # 处理来自 programmatic_config_override 的其他 TrainingArguments 字段覆盖
    if programmatic_config_override:
        for key, value in programmatic_config_override.items():
            if key not in ['model_constructor_params', 'run_name']: # 这些已特殊处理或将特殊处理
                # 简单假设这些是 TrainingArguments 的直接字段
                train_args_constructor_kwargs[key] = value
                
    training_args = TrainingArguments.from_model_name(
        model_name=model_name,
        **train_args_constructor_kwargs
    )

    # 应用 programmatic_config_override 中的 model_constructor_params
    if programmatic_config_override and 'model_constructor_params' in programmatic_config_override:
        if not hasattr(training_args, 'model_constructor_params') or training_args.model_constructor_params is None:
            training_args.model_constructor_params = {}
        
        for key, value in programmatic_config_override['model_constructor_params'].items():
            training_args.model_constructor_params[key] = value
            
    train_dataset, eval_dataset = get_cifar100_datasets(
        use_imagenet_norm=getattr(training_args, 'use_imagenet_norm', False)
    )
    
    num_classes = get_num_classes('cifar100')
    
    model_constructor_params_to_use = getattr(training_args, 'model_constructor_params', {})
    model = get_model(model_name, num_classes=num_classes, **model_constructor_params_to_use)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if trainer.rank == 0:
        print(f"开始训练模型: {model_name}")
        
        run_label_suffix = ""
        if programmatic_config_override:
            override_details = []
            if 'model_constructor_params' in programmatic_config_override:
                 mcp_string = ", ".join(f"{k}={v}" for k,v in programmatic_config_override['model_constructor_params'].items())
                 if mcp_string: override_details.append(f"model_params:{{{mcp_string}}}")
            
            other_overrides_string = ", ".join(f"{k}={v}" for k,v in programmatic_config_override.items() if k not in ['run_name', 'model_constructor_params'])
            if other_overrides_string: override_details.append(f"args:{{{other_overrides_string}}}")
            
            if override_details:
                run_label_suffix = f" (Overrides: {'; '.join(override_details)})"
        
        effective_run_name = training_args.run_name if training_args.run_name else model_name
        print(f"运行标签: {effective_run_name}{run_label_suffix}")
        print(f"训练参数: {training_args}")
        
        model_to_inspect = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_info_str = get_model_info(model_to_inspect, model_name) 
        print(f"模型信息: {model_info_str}")

    trainer.train()

    if trainer.rank == 0:
        final_run_name = training_args.run_name if training_args.run_name else model_name
        log_dir_path = training_args.output_dir
        print(f"模型 {model_name}{run_label_suffix} 训练完成。日志保存在: {log_dir_path}")


def main():
    parser = argparse.ArgumentParser(description="通用CIFAR-100训练脚本")
    parser.add_argument("--model_name", type=str, required=True, help="要训练的模型名称 (例如: resnet_56, coatnet_0, coatnet_cifar_opt)")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次 (覆盖模型默认设置)")
    parser.add_argument("--batch_size", type=int, default=None, help="每个设备的批次大小 (覆盖模型默认设置)")
    parser.add_argument("--lr", type=float, default=None, help="学习率 (覆盖模型默认设置)")
    parser.add_argument("--output_dir", type=str, default=None, help="基础日志和模型保存目录 (例如 ./logs), 最终会是 ./logs/run_name")
    parser.add_argument("--run_name", type=str, default=None, help="为本次运行指定一个名称 (用于日志子目录名)")
    
    args = parser.parse_args()

    cli_overrides = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'output_dir': args.output_dir, # Pass base output_dir
        'run_name': args.run_name
    }
    cli_overrides_filtered = {k: v for k, v in cli_overrides.items() if v is not None}
    
    # 检查环境变量中的 model_constructor_params
    programmatic_override = {}
    if 'MODEL_CONSTRUCTOR_PARAMS' in os.environ:
        try:
            model_constructor_params = json.loads(os.environ['MODEL_CONSTRUCTOR_PARAMS'])
            programmatic_override['model_constructor_params'] = model_constructor_params
            print(f"从环境变量获取模型构造参数: {model_constructor_params}")
        except json.JSONDecodeError as e:
            print(f"警告: 无法解析环境变量 MODEL_CONSTRUCTOR_PARAMS: {e}")
    
    run_training_config(
        model_name=args.model_name, 
        cli_args_dict=cli_overrides_filtered,
        programmatic_config_override=programmatic_override
    )

if __name__ == "__main__":
    main() 