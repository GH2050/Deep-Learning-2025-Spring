import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
import time
import os
from pathlib import Path
import json

from model import MODEL_REGISTRY, get_model_info

class CIFAR100Trainer:
    def __init__(self, model_name, epochs=15, batch_size=128, learning_rate=0.1):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 初始化accelerator
        self.accelerator = Accelerator()
        
        # 创建数据加载器
        self.train_loader, self.test_loader = self._create_data_loaders()
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建优化器和调度器
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用accelerator准备所有组件
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.test_loader
        )
        
    def _create_data_loaders(self):
        """创建数据加载器"""
        # 训练数据变换
        transform_train = transforms.Compose([
            transforms.Resize(224),  # 为了适配预训练模型
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 测试数据变换
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 加载CIFAR-100数据集
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def _create_model(self):
        """创建模型"""
        from model import get_model
        return get_model(self.model_name)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                self.accelerator.print(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Acc: {100.*correct/total:.2f}%'
                )
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def test(self):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Top-5准确率
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 += top5_pred.eq(targets.view(1, -1).expand_as(top5_pred)).sum().item()
        
        test_loss /= len(self.test_loader)
        top1_acc = 100. * correct / total
        top5_acc = 100. * correct_top5 / total
        
        return test_loss, top1_acc, top5_acc
    
    def train(self):
        """完整训练过程"""
        best_acc = 0
        results = {
            'model_name': self.model_name,
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_acc_top5': [],
            'best_acc': 0,
            'best_acc_top5': 0,
            'total_time': 0,
            'parameters': 0
        }
        
        # 计算参数量
        model_info = get_model_info(self.model_name)
        results['parameters'] = model_info['parameters_M']
        
        self.accelerator.print(f"开始训练模型: {self.model_name}")
        self.accelerator.print(f"参数量: {results['parameters']:.2f}M")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 测试
            test_loss, test_acc, test_acc_top5 = self.test()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录结果
            results['epochs'].append(epoch + 1)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            results['test_acc_top5'].append(test_acc_top5)
            
            if test_acc > best_acc:
                best_acc = test_acc
                results['best_acc'] = best_acc
                results['best_acc_top5'] = test_acc_top5
                
                # 保存最佳模型
                self.save_model(epoch + 1, test_acc)
            
            epoch_time = time.time() - epoch_start
            
            self.accelerator.print(
                f'Epoch {epoch+1}/{self.epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                f'Test Top5: {test_acc_top5:.2f}%, Time: {epoch_time:.1f}s'
            )
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        self.accelerator.print(f"训练完成! 最佳准确率: {best_acc:.2f}%, 总时间: {total_time:.1f}s")
        
        # 保存训练结果
        self.save_results(results)
        
        return results
    
    def save_model(self, epoch, acc):
        """保存模型检查点"""
        if self.accelerator.is_main_process:
            checkpoint_dir = Path('logs/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': acc,
                'model_name': self.model_name
            }
            
            checkpoint_path = checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, checkpoint_path)
    
    def save_results(self, results):
        """保存训练结果"""
        if self.accelerator.is_main_process:
            results_dir = Path('logs/results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / f'{self.model_name}_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

def train_all_models():
    """训练所有模型"""
    
    # 要训练的模型列表
    models_to_train = [
        'resnet_20',                # 基础ResNet
        'eca_resnet_20',           # ECA注意力ResNet
        'ghost_resnet_20',         # Ghost轻量化ResNet
        'convnext_tiny',           # ConvNeXt-Tiny
        'convnext_tiny_timm',      # timm版ConvNeXt
        'segnext_mscan_tiny',      # SegNeXt MSCA
        'coatnet_0',               # CoAtNet
        'cspresnet50',             # CSPNet
        'ghostnet_100',            # GhostNet
        'hornet_tiny',             # HorNet
        'resnest50d',              # ResNeSt
        'mlp_mixer_tiny',          # MLP-Mixer自实现
        'mlp_mixer_b16',           # timm版MLP-Mixer
    ]
    
    all_results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建训练器
            trainer = CIFAR100Trainer(
                model_name=model_name,
                epochs=15,
                batch_size=128,
                learning_rate=0.1
            )
            
            # 训练模型
            results = trainer.train()
            all_results[model_name] = results
            
            print(f"模型 {model_name} 训练完成!")
            print(f"最佳准确率: {results['best_acc']:.2f}%")
            print(f"参数量: {results['parameters']:.2f}M")
            print(f"训练时间: {results['total_time']:.1f}s")
            
        except Exception as e:
            print(f"模型 {model_name} 训练失败: {e}")
            all_results[model_name] = {'error': str(e)}
            continue
    
    # 保存所有结果的汇总
    summary_path = Path('logs/results/all_models_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print("所有模型训练结果汇总:")
    print(f"{'='*80}")
    print(f"{'模型名称':<20} {'最佳准确率':<10} {'Top5准确率':<10} {'参数量(M)':<10} {'训练时间(s)':<12}")
    print("-" * 80)
    
    for model_name, result in all_results.items():
        if 'error' not in result:
            print(f"{model_name:<20} {result['best_acc']:<10.2f} {result['best_acc_top5']:<10.2f} "
                  f"{result['parameters']:<10.2f} {result['total_time']:<12.1f}")
        else:
            print(f"{model_name:<20} {'失败':<10} {'失败':<10} {'失败':<10} {'失败':<12}")

if __name__ == "__main__":
    train_all_models() 