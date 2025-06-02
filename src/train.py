#!/usr/bin/env python3
import sys
import argparse
import os
import torch

sys.path.append('src')
from model import get_model, get_model_info
from dataset import get_cifar100_datasets
from trainer import Trainer, TrainingArguments
from utils import get_num_classes # 假设utils.py中有这个函数

def main():
    parser = argparse.ArgumentParser(description="通用CIFAR-100训练脚本")
    parser.add_argument("--model_name", type=str, required=True, help="要训练的模型名称 (例如: resnet_56, coatnet_0, coatnet_cifar_opt)")
    parser.add_argument("--epochs", type=int, help="训练轮次 (覆盖模型默认设置)")
    parser.add_argument("--batch_size", type=int, help="每个设备的批次大小 (覆盖模型默认设置)")
    parser.add_argument("--lr", type=float, help="学习率 (覆盖模型默认设置)")
    parser.add_argument("--output_dir", type=str, default="./logs", help="日志和模型保存目录")
    parser.add_argument("--run_name", type=str, help="为本次运行指定一个名称 (用于日志子目录)")
    # 可以从TrainingArguments中获取更多可配置的参数，或者直接让用户通过 TrainingArguments.from_model_name 获取

    args = parser.parse_args()

    # 构造传递给 TrainingArguments.from_model_name 的kwargs
    # 确保只传递用户明确指定的参数，以便hparams中的默认值可以生效
    train_args_kwargs = {}
    if args.epochs is not None:
        train_args_kwargs['num_train_epochs'] = args.epochs
    if args.batch_size is not None:
        train_args_kwargs['per_device_train_batch_size'] = args.batch_size
        train_args_kwargs['per_device_eval_batch_size'] = args.batch_size * 2 # 假设评估时批次加倍
    if args.lr is not None:
        train_args_kwargs['learning_rate'] = args.lr
    if args.output_dir:
         # output_dir 将由 TrainingArguments 内部处理，结合 model_name 和 run_name
        train_args_kwargs['output_dir'] = args.output_dir
    if args.run_name:
        train_args_kwargs['run_name'] = args.run_name


    # 使用 model_name 获取训练参数，并用命令行参数覆盖
    training_args = TrainingArguments.from_model_name(
        model_name=args.model_name,
        **train_args_kwargs
    )

    # 获取数据集
    # 假设 get_cifar100_datasets 可以接受 training_args 中的某些参数，比如 use_imagenet_norm
    train_dataset, eval_dataset = get_cifar100_datasets(
        use_imagenet_norm=getattr(training_args, 'use_imagenet_norm', False)
    )
    
    num_classes = get_num_classes('cifar100') # 获取类别数

    # 获取模型
    # 假设 get_model 可以接受 training_args 中的 model_constructor_params
    model_constructor_params = getattr(training_args, 'model_constructor_params', {})
    model = get_model(args.model_name, num_classes=num_classes, **model_constructor_params)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics 可以根据需要添加
    )

    # 开始训练
    if trainer.rank == 0: # 主进程打印信息
        print(f"开始训练模型: {args.model_name}")
        print(f"训练参数: {training_args}")
        model_info = get_model_info(trainer.model if not trainer.distributed else trainer.model.module, args.model_name) # 获取原始模型信息
        print(f"模型信息: {model_info}")

    trainer.train()

    if trainer.rank == 0:
        print(f"模型 {args.model_name} 训练完成。日志保存在: {training_args.output_dir}")

if __name__ == "__main__":
    main() 