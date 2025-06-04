import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader100 import trainloader,testloader,classes
import os
import csv

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks_list, num_classes=100, in_channels=3, block_kwargs=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes

        if block_kwargs is None:
            block_kwargs = {}

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        planes_list = [16, 32, 64]
        strides = [1, 2, 2]

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_list):
            stage_planes = planes_list[i]
            stage_stride = strides[i]
            self.layers.append(self._make_layer(block_type, stage_planes, num_blocks, stride=stage_stride, block_kwargs=block_kwargs))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = planes_list[-1] * block_type.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block_type, planes, num_blocks, stride, block_kwargs):
        strides_list = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides_list:
            layers.append(block_type(self.in_planes, planes, s, **block_kwargs))
            self.in_planes = planes * block_type.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer_module in self.layers:
            out = layer_module(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        logits = self.fc(features)
        if return_feature:
            return logits, features
        return logits

def newmodel3_builder(num_classes=100, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, block_kwargs=kwargs)

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers, 2).sum(dim=1) - \
                  2 * torch.matmul(features, self.centers.t())
        labels = labels.view(-1, 1)
        mask = torch.zeros_like(distmat)
        mask.scatter_(1, labels, 1)
        loss = (distmat * mask).sum() / batch_size
        return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = newmodel3_builder(num_classes=100).to(device)
criterion_ce = nn.CrossEntropyLoss()
criterion_center = CenterLoss(num_classes=100, feat_dim=net.feature_dim, device=device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.5)



def training(log_path):
    # 打开CSV文件并写入表头
    with open(log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "loss"])

    net.train()
    
    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            outputs, features = net(inputs, return_feature=True)
            loss_ce = criterion_ce(outputs, labels)
            loss_center = criterion_center(features, labels)

            loss = loss_ce + 0.01 * loss_center
            loss.backward()

            optimizer.step()
            optimizer_center.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            with open(log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    print('Finished Training')

def evaluate_model(model, testloader, classes, save_path,device):
    """
    通用模型评估函数。
    
    Args:
        model (torch.nn.Module): 训练好的模型
        testloader (DataLoader): 测试数据加载器
        classes (list[str], optional): 类别名列表；若提供则可输出每类准确率
        show_per_class (bool): 是否显示每个类别的准确率
        device (str): 模型使用的设备（'cpu' or 'cuda'）
        
    Prints:
        - Overall accuracy
        - (Optional) Per-class accuracy
    """
    model.eval()
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    results={}
    # whole_acc
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    overall_acc = 100 * correct / total
    print(f" Overall Accuracy: {overall_acc:.2f} %")
    results["overall_accuracy"] = round(overall_acc, 2)

    # acc for each class
    print(" Per-Class Accuracy:")
    class_acc={}
    for classname in classes:
        if total_pred[classname] == 0:
            acc = 0.0
        else:
            acc = 100 * correct_pred[classname] / total_pred[classname]
        class_acc[classname] = round(acc, 2)
        print(f"  - {classname:15s}: {acc:.1f} %")
    results["per_class_accuracy"] = class_acc

    # save to file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Accuracy (%)"])
            for classname, acc in results["per_class_accuracy"].items():
                writer.writerow([classname, acc])
            writer.writerow([])
            writer.writerow(["Overall Accuracy", results["overall_accuracy"]])
        print(f"Results saved to CSV: {save_path}")


if __name__=='__main__':

    log_path = './GhostNet/result/log_newnet3_100epoch.csv'
    training(log_path)
    PATH = './GhostNet/models/NewNet3_model_10.pth'
    torch.save(net.state_dict(), PATH)

    result_path='./GhostNet/result/CIFAR100_newnet3_result_10epoch.csv'
    evaluate_model(net, testloader, classes,result_path,device=device)
