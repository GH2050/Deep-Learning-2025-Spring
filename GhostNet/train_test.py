import torch.optim as optim
import torch.nn as nn
import torch
import csv
import os

from GhostNet_model import ghost_resnet20_builder as GhostNet20
from GhostNet_model import ghost_resnet32_builder as GhostNet32
from ECANet_model import ecanet20_builder as ECANet20
from dataloader import trainloader,testloader,classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ECA-Net 模型测试
net = ECANet20(num_classes=100, use_eca=True, adaptive_kernel=True)
net = ECANet20(num_classes=100, use_eca=True, adaptive_kernel=False, fixed_kernel_size=3)
net = ECANet20(num_classes=100, use_eca=False, adaptive_kernel=True)
net = ECANet20(num_classes=100, use_eca=False, adaptive_kernel=False, fixed_kernel_size=3)

# net = GhostNet20()
#net = GhostNet32()
net.to(device)

# (1)loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# （2）train
def training():
    net.train()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
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

    training()
    PATH = './GhostNet/GhostNet_model.pth'
    torch.save(net.state_dict(), PATH)

    # net = GhostNet20()
    # net.load_state_dict(torch.load(PATH))
    # net.to(device)

    result_path='./GhostNet/result/ghost_result.csv'
    evaluate_model(net, testloader, classes,result_path,device=device,)


