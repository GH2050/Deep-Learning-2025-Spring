#!/usr/bin/env python3
import os
import json
import datetime
import subprocess
import glob
from pathlib import Path

def get_latest_training_results():
    """获取最新的训练结果"""
    results = {}
    
    # 查找最新的训练结果JSON文件
    json_files = glob.glob('logs/training_results_*.json')
    if json_files:
        latest_json = max(json_files, key=os.path.getctime)
        with open(latest_json, 'r') as f:
            training_session = json.load(f)
            
        for model_name, model_result in training_session.items():
            if model_result['success']:
                # 查找对应的历史文件
                history_files = glob.glob(f'logs/{model_name}_history_*.json')
                if history_files:
                    latest_history = max(history_files, key=os.path.getctime)
                    with open(latest_history, 'r') as f:
                        history = json.load(f)
                    
                    results[model_name] = {
                        'status': '✅ 已完成',
                        'best_acc': max(history['test_acc']),
                        'final_acc': history['test_acc'][-1],
                        'epochs': f"{len(history['test_acc'])}/{model_result['epochs']}",
                        'training_time': f"~{model_result['training_time']/60:.0f}分钟"
                    }
                else:
                    results[model_name] = {
                        'status': '✅ 已完成',
                        'best_acc': 0,
                        'final_acc': 0,
                        'epochs': f"?/{model_result['epochs']}",
                        'training_time': f"~{model_result['training_time']/60:.0f}分钟"
                    }
            else:
                results[model_name] = {
                    'status': '❌ 训练失败',
                    'best_acc': 0,
                    'final_acc': 0,
                    'epochs': f"0/{model_result['epochs']}",
                    'training_time': '-'
                }
    
    # 如果没有找到完整的训练会话，查找单独的历史文件
    if not results:
        for model_name in ['resnet_20', 'eca_resnet_20', 'ghost_resnet_20', 'convnext_tiny']:
            history_files = glob.glob(f'logs/{model_name}_history_*.json')
            if history_files:
                latest_history = max(history_files, key=os.path.getctime)
                with open(latest_history, 'r') as f:
                    history = json.load(f)
                
                results[model_name] = {
                    'status': '✅ 已完成',
                    'best_acc': max(history['test_acc']),
                    'final_acc': history['test_acc'][-1],
                    'epochs': f"{len(history['test_acc'])}/20",
                    'training_time': f"~{sum(history['epoch_times'])/60:.0f}分钟"
                }
            else:
                results[model_name] = {
                    'status': '⏳ 待训练',
                    'best_acc': 0,
                    'final_acc': 0,
                    'epochs': '-',
                    'training_time': '-'
                }
    
    return results

def get_model_info():
    """获取模型参数量信息"""
    model_info = {
        'resnet_20': {'params': '0.28M', 'name': 'ResNet-20'},
        'eca_resnet_20': {'params': '0.28M', 'name': 'ECA-ResNet-20'},
        'ghost_resnet_20': {'params': '0.03M', 'name': 'Ghost-ResNet-20'},
        'convnext_tiny': {'params': '0.17M', 'name': 'ConvNeXt-Tiny'}
    }
    return model_info

def update_readme_progress(results):
    """更新README.md中的训练进展表格"""
    
    # 读取现有README
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 模型信息
    model_info = get_model_info()
    
    # 生成新的表格
    table_lines = [
        "| 模型 | 参数量 | 训练状态 | 最佳准确率 | 训练轮次 | 训练时间 |",
        "|------|--------|----------|------------|----------|----------|"
    ]
    
    for model_key, result in results.items():
        model_name = model_info.get(model_key, {}).get('name', model_key)
        params = model_info.get(model_key, {}).get('params', 'Unknown')
        status = result['status']
        best_acc = f"{result['best_acc']:.2f}%" if result['best_acc'] > 0 else "-"
        epochs = result['epochs']
        training_time = result['training_time']
        
        table_lines.append(f"| {model_name} | {params} | {status} | {best_acc} | {epochs} | {training_time} |")
    
    new_table = '\n'.join(table_lines)
    
    # 替换表格
    import re
    pattern = r'(\| 模型 \| 参数量.*?\n\|---.*?\n)(.*?)(\n\n###)'
    
    def replace_table(match):
        return match.group(1) + new_table.replace(table_lines[0] + '\n' + table_lines[1] + '\n', '') + match.group(3)
    
    new_content = re.sub(pattern, replace_table, content, flags=re.DOTALL)
    
    # 更新最后更新时间
    timestamp = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
    new_content = re.sub(
        r'(\*\*最后更新\*\*:).*?(\n)',
        f'\\1 {timestamp}\\2',
        new_content
    )
    
    # 如果没有找到最后更新行，则在表格后添加
    if '**最后更新**:' not in new_content:
        new_content = re.sub(
            r'(\n\n### 详细训练记录)',
            f'\n\n**最后更新**: {timestamp}\n\\1',
            new_content
        )
    
    # 写回文件
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return new_content

def git_commit_changes():
    """进行git提交"""
    try:
        # 添加文件
        subprocess.run(['git', 'add', 'README.md'], check=True)
        subprocess.run(['git', 'add', 'logs/'], check=True)
        
        # 生成提交信息
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"更新训练进展 - {timestamp}"
        
        # 提交
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # 推送（如果配置了远程仓库）
        try:
            subprocess.run(['git', 'push'], check=True)
            return True, "Git提交和推送成功"
        except subprocess.CalledProcessError:
            return True, "Git提交成功，但推送失败（可能没有配置远程仓库）"
            
    except subprocess.CalledProcessError as e:
        return False, f"Git操作失败: {e}"

def main():
    """主函数"""
    print("开始更新训练进展...")
    
    # 获取训练结果
    results = get_latest_training_results()
    
    if not results:
        print("未找到训练结果，跳过更新")
        return
    
    print("找到以下训练结果:")
    for model, result in results.items():
        print(f"  {model}: {result['status']}")
    
    # 更新README
    print("\n更新README.md...")
    try:
        update_readme_progress(results)
        print("✅ README.md更新成功")
    except Exception as e:
        print(f"❌ README.md更新失败: {e}")
        return
    
    # Git提交
    print("\n进行Git提交...")
    success, message = git_commit_changes()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    print("\n进展更新完成!")

if __name__ == "__main__":
    main() 