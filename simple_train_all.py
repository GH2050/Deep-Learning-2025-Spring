#!/usr/bin/env python3

import os
import time
import subprocess
import sys

def run_single_model(model_name, epochs=15):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "src/train_accelerate.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", "128",
        "--lr", "0.1"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        success = True
        print(f"✅ {model_name} 训练成功完成!")
    except subprocess.CalledProcessError as e:
        success = False
        print(f"❌ {model_name} 训练失败: {e}")
    except KeyboardInterrupt:
        print(f"\n⚠️ {model_name} 训练被用户中断")
        return False
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练耗时: {training_time/60:.1f}分钟")
    
    return success

def main():
    """主训练流程"""
    models = [
        'resnet_20',
        'eca_resnet_20', 
        'ghost_resnet_20',
        'convnext_tiny'
    ]
    
    print("CIFAR-100 多模型训练开始")
    print(f"计划训练模型: {', '.join(models)}")
    
    results = {}
    total_start = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\n进度: {i}/{len(models)}")
        success = run_single_model(model, epochs=15)
        results[model] = success
        
        if not success:
            print(f"模型 {model} 失败，继续下一个...")
        
        # 训练间隔
        if i < len(models):
            print("等待5秒...")
            time.sleep(5)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("训练完成汇总")
    print(f"{'='*60}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    
    success_count = sum(results.values())
    print(f"成功: {success_count}/{len(models)} 个模型")
    
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {model}: {status}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1) 