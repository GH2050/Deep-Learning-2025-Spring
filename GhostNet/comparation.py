import torch
import csv
import os
from ptflops import get_model_complexity_info

from GhostNet_model import ghost_resnet20_builder as GhostNet
from ResNet_model import resnet20_builder as ResNet

input_shape = (3, 32, 32)
csv_path = './GhostNet/result/model_comparison_results.csv'

ghostnet = GhostNet(num_classes=100)
resnet = ResNet(num_classes=100)

# 获取模型统计信息 
def get_stats(model, input_res):
    model.eval()
    with torch.cuda.device(0 if torch.cuda.is_available() else -1):
        macs, params = get_model_complexity_info(
            model, input_res, as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
    return macs, params

ghostnet_macs, ghostnet_params = get_stats(ghostnet, input_shape)
resnet_macs, resnet_params = get_stats(resnet, input_shape)

# 写入CSV文件 
os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Parameters', 'FLOPs'])
    writer.writerow(['GhostNet-20', ghostnet_params, ghostnet_macs])
    writer.writerow(['ResNet-20', resnet_params, resnet_macs])

