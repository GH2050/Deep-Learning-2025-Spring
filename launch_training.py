#!/usr/bin/env python3

import subprocess
import sys
import os
import datetime
import time
import json

def run_training(model_name, epochs=20, batch_size=128, lr=0.1):
    """直接启动单个模型的训练"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "src/train_accelerate.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        success = True
        error_msg = None
        print("训练成功完成!")
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        print(f"训练失败: {error_msg}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return success, training_time, error_msg

def log_overall_progress(log_file, message):
    """记录总体训练进度"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

def main():
    """主训练流程"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 总体日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_log = f'logs/training_session_{timestamp}.log'
    
    # 初始化日志
    with open(overall_log, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIFAR-100 多模型训练会话开始\n")
        f.write("="*80 + "\n")
        f.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python版本: {sys.version}\n")
        f.write("="*80 + "\n\n")
    
    # 训练配置
    models_to_train = [
        ('resnet_20', 20, 128, 0.1),
        ('eca_resnet_20', 20, 128, 0.1),
        ('ghost_resnet_20', 20, 128, 0.1),
        ('convnext_tiny', 20, 128, 0.1)
    ]
    
    results = {}
    total_start_time = time.time()
    
    log_overall_progress(overall_log, f"计划训练 {len(models_to_train)} 个模型")
    
    for i, (model_name, epochs, batch_size, lr) in enumerate(models_to_train, 1):
        log_overall_progress(overall_log, f"({i}/{len(models_to_train)}) 开始训练 {model_name}")
        log_overall_progress(overall_log, f"参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        
        success, training_time, error_msg = run_training(model_name, epochs, batch_size, lr)
        
        results[model_name] = {
            'success': success,
            'training_time': training_time,
            'error_msg': error_msg,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr
        }
        
        if success:
            log_overall_progress(overall_log, f"✅ {model_name} 训练完成，耗时: {training_time/60:.1f}分钟")
        else:
            log_overall_progress(overall_log, f"❌ {model_name} 训练失败")
            if error_msg:
                log_overall_progress(overall_log, f"错误信息: {error_msg}")
        
        # 训练间隔，让系统休息一下
        if i < len(models_to_train):
            log_overall_progress(overall_log, "等待5秒后开始下一个模型训练...")
            time.sleep(5)
    
    total_time = time.time() - total_start_time
    
    # 生成最终报告
    with open(overall_log, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("训练会话完成\n")
        f.write("="*80 + "\n")
        f.write(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time/60:.1f}分钟\n\n")
        
        f.write("训练结果汇总:\n")
        f.write("-"*50 + "\n")
        for model, result in results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            f.write(f"{model:<20} {status:<8} {result['training_time']/60:.1f}分钟\n")
        
        successful_models = [m for m, r in results.items() if r['success']]
        f.write(f"\n成功训练模型数: {len(successful_models)}/{len(models_to_train)}\n")
        f.write("="*80 + "\n")
    
    # 保存结果到JSON
    results_file = f'logs/training_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_overall_progress(overall_log, f"训练会话完成! 总耗时: {total_time/60:.1f}分钟")
    log_overall_progress(overall_log, f"成功训练: {len(successful_models)}/{len(models_to_train)} 个模型")
    log_overall_progress(overall_log, f"详细日志: {overall_log}")
    log_overall_progress(overall_log, f"结果文件: {results_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        # 如果所有模型都训练成功，退出码为0
        success_count = sum(1 for r in results.values() if r['success'])
        if success_count == len(results):
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n训练过程发生错误: {e}")
        sys.exit(1) 