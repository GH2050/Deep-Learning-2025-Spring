#!/usr/bin/env python3
import sys
import argparse

sys.path.append('src')
from model import get_model
from dataset import get_cifar100_datasets
from trainer import Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser(description='CIFAR-100模型训练')
    parser.add_argument('--model_name', type=str, default='resnet_56', help='模型名称')
    parser.add_argument('--output_dir', type=str, default='./logs', help='输出目录')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (可选, 若不提供则使用模型在utils.py中定义的默认值)')
    
    args = parser.parse_args()
    
    training_args_params = {
        "model_name": args.model_name,
        "output_dir": f"{args.output_dir}/{args.model_name}",
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "use_mixup": True, # Default from previous setup
        "label_smoothing_factor": 0.1 # Default from previous setup
    }

    if args.lr is not None:
        training_args_params["learning_rate"] = args.lr
        
    training_args = TrainingArguments.from_model_name(**training_args_params)
    
    model = get_model(
        args.model_name, 
        num_classes=100,
        **training_args.model_constructor_params
    )
    
    train_dataset, eval_dataset = get_cifar100_datasets(
        data_dir='./data',
        use_imagenet_norm=training_args.use_imagenet_norm
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()

if __name__ == '__main__':
    main() 