# models/eca.py
import math
import torch
import torch.nn as nn

class ECABlock(nn.Module):
    """
    ECA 注意力模块：对输入特征进行 GAP 后，使用 1D 卷积学习通道权重。
    参考：Wang 等人，ECA-Net (CVPR 2020):contentReference[oaicite:2]{index=2}。
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 根据通道数自适应确定 1D 卷积核大小
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积：输入通道 1，输出通道 1，kernel_size=k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, 
                              padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B x C x H x W
        y = self.avg_pool(x)            # B x C x 1 x 1
        y = y.view(y.size(0), y.size(1))# B x C
        y = y.unsqueeze(1)             # B x 1 x C
        y = self.conv(y)               # B x 1 x C
        y = y.view(x.size(0), x.size(1), 1, 1)  # B x C x 1 x 1
        y = self.sigmoid(y)
        return x * y.expand_as(x)      # 通道乘以注意力权重
