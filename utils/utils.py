# utils/utils.py
import torch

def accuracy(output, target):
    """
    计算输出output与真实标签target的匹配数量（正确预测个数）。
    """
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
    return correct
