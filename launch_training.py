#!/usr/bin/env python3

import subprocess
import sys
import os

def launch_accelerate_training():
    config_file = "default_config.yaml"
    
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在!")
        return
    
    models = ['resnet_20', 'eca_resnet_20']
    epochs = 25
    batch_size = 256
    lr = 0.1
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"启动 {model} 加速训练")
        print(f"{'='*60}")
        
        cmd = [
            "accelerate", "launch",
            "--config_file", config_file,
            "src/train_accelerate.py",
            "--model", model,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr)
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"{model} 训练完成!")
        except subprocess.CalledProcessError as e:
            print(f"{model} 训练失败: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n训练被用户中断")
            break
    
    print("\n所有模型训练完成!")

if __name__ == "__main__":
    launch_accelerate_training() 