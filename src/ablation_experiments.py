import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
import json
import time
from pathlib import Path
import numpy as np

from model import ResNet, BasicBlock, ECABasicBlock, ECALayer, GhostBottleneck, get_model_info

class AblationExperiment:
    def __init__(self, experiment_name, epochs=10, batch_size=128, learning_rate=0.1):
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.accelerator = Accelerator()
        self.train_loader, self.test_loader = self._create_data_loaders()
        
    def _create_data_loaders(self):
        """创建数据加载器"""
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def train_model(self, model, model_name):
        """训练模型"""
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        
        model, optimizer, train_loader, test_loader = self.accelerator.prepare(
            model, optimizer, self.train_loader, self.test_loader
        )
        
        results = {
            'model_name': model_name,
            'experiment': self.experiment_name,
            'epochs': [],
            'train_acc': [],
            'test_acc': [],
            'test_acc_top5': [],
            'best_acc': 0,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        }
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # 训练
            model.train()
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                self.accelerator.backward(loss)
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # 测试
            model.eval()
            test_correct = 0
            test_correct_top5 = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
                    
                    _, top5_pred = outputs.topk(5, 1, True, True)
                    top5_pred = top5_pred.t()
                    test_correct_top5 += top5_pred.eq(targets.view(1, -1).expand_as(top5_pred)).sum().item()
            
            test_acc = 100. * test_correct / test_total
            test_acc_top5 = 100. * test_correct_top5 / test_total
            
            results['epochs'].append(epoch + 1)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            results['test_acc_top5'].append(test_acc_top5)
            
            if test_acc > results['best_acc']:
                results['best_acc'] = test_acc
            
            scheduler.step()
            
            self.accelerator.print(
                f'Epoch {epoch+1}/{self.epochs}: '
                f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%'
            )
        
        results['total_time'] = time.time() - start_time
        return results

class ECANetAblation:
    """ECA-Net消融实验"""
    
    @staticmethod
    def get_models():
        """获取消融实验的模型"""
        models = {}
        
        # 1. 基础ResNet
        models['baseline'] = ResNet(BasicBlock, [3, 3, 3])
        
        # 2. ResNet + ECA
        models['with_eca'] = ResNet(ECABasicBlock, [3, 3, 3])
        
        # 3. 不同ECA核大小的ResNet
        class ECABasicBlockK3(nn.Module):
            expansion = 1
            
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                
                # 固定核大小为3的ECA
                self.eca = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid()
                )
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                
                # ECA attention
                y = self.eca[0](out)  # AdaptiveAvgPool2d
                y = self.eca[1](y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
                y = self.eca[2](y)  # Sigmoid
                out = out * y.expand_as(out)
                
                out += self.shortcut(x)
                out = torch.relu(out)
                return out
        
        models['eca_k3'] = ResNet(ECABasicBlockK3, [3, 3, 3])
        
        return models

class GhostNetAblation:
    """GhostNet消融实验"""
    
    @staticmethod
    def get_models():
        """获取Ghost消融实验的模型"""
        models = {}
        
        # 1. 基础ResNet
        models['baseline'] = ResNet(BasicBlock, [3, 3, 3])
        
        # 2. GhostNet ResNet
        models['ghost'] = ResNet(GhostBottleneck, [3, 3, 3])
        
        # 3. 不同ratio的GhostNet
        class GhostBottleneckRatio4(nn.Module):
            expansion = 1
            
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                from model import GhostModule
                
                hidden_dim = in_planes
                self.ghost1 = GhostModule(in_planes, hidden_dim, kernel_size=1, ratio=4, relu=True)
                
                if stride > 1:
                    self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
                    self.bn_dw = nn.BatchNorm2d(hidden_dim)
                
                self.ghost2 = GhostModule(hidden_dim, planes, kernel_size=1, ratio=4, relu=False)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, 1, stride, 0, bias=False),
                        nn.BatchNorm2d(planes),
                    )
                
                self.stride = stride
            
            def forward(self, x):
                residual = x
                x = self.ghost1(x)
                if self.stride > 1:
                    x = self.bn_dw(self.conv_dw(x))
                x = self.ghost2(x)
                x += self.shortcut(residual)
                return x
        
        models['ghost_ratio4'] = ResNet(GhostBottleneckRatio4, [3, 3, 3])
        
        return models

class AttentionPositionAblation:
    """注意力位置消融实验"""
    
    @staticmethod
    def get_models():
        models = {}
        
        # 1. 基础ResNet
        models['baseline'] = ResNet(BasicBlock, [3, 3, 3])
        
        # 2. 注意力在残差连接之前
        class ECABeforeResidual(nn.Module):
            expansion = 1
            
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.eca = ECALayer(planes)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.eca(out)  # 在残差连接之前
                out += self.shortcut(x)
                out = torch.relu(out)
                return out
        
        # 3. 注意力在残差连接之后
        class ECAAfterResidual(nn.Module):
            expansion = 1
            
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.eca = ECALayer(planes)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.eca(out)  # 在残差连接之后
                out = torch.relu(out)
                return out
        
        models['eca_before_residual'] = ResNet(ECABeforeResidual, [3, 3, 3])
        models['eca_after_residual'] = ResNet(ECAAfterResidual, [3, 3, 3])
        
        return models

def run_ablation_experiments():
    """运行所有消融实验"""
    
    experiments = [
        ("ECA-Net消融实验", ECANetAblation),
        ("GhostNet消融实验", GhostNetAblation),
        ("注意力位置消融实验", AttentionPositionAblation)
    ]
    
    all_results = {}
    
    for exp_name, exp_class in experiments:
        print(f"\n{'='*60}")
        print(f"开始{exp_name}")
        print(f"{'='*60}")
        
        experiment = AblationExperiment(exp_name, epochs=10)
        models = exp_class.get_models()
        exp_results = {}
        
        for model_name, model in models.items():
            print(f"\n训练模型: {model_name}")
            try:
                result = experiment.train_model(model, f"{exp_name}_{model_name}")
                exp_results[model_name] = result
                print(f"最佳准确率: {result['best_acc']:.2f}%")
            except Exception as e:
                print(f"模型 {model_name} 训练失败: {e}")
                exp_results[model_name] = {'error': str(e)}
        
        all_results[exp_name] = exp_results
        
        # 保存实验结果
        results_dir = Path('logs/ablation_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        exp_file = results_dir / f'{exp_name.replace(" ", "_")}.json'
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(exp_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{exp_name}结果汇总:")
        print("-" * 50)
        for model_name, result in exp_results.items():
            if 'error' not in result:
                print(f"{model_name:<20}: {result['best_acc']:6.2f}% (参数: {result['parameters']:.2f}M)")
            else:
                print(f"{model_name:<20}: 失败")
    
    # 保存总结果
    summary_file = Path('logs/ablation_results/all_ablation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results

if __name__ == "__main__":
    results = run_ablation_experiments()
    print("\n所有消融实验完成!") 