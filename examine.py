import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network import SetNet

# 创建一个虚拟的输入数据
batch_size = 2
frame_num = 5
keypoints_num = 16
channel = 1  # 灰度图，通道数为1
hidden_dim = 256  # 隐藏层维度

# 生成随机的 silho 输入数据（示例）
silho = torch.rand(batch_size, frame_num, keypoints_num, channel)

# 创建 SetNet 模型
model = SetNet(hidden_dim)

# 打印模型结构
print(model)

# 运行模型的前向传播
feature, _ = model(silho)

# 检查输出的形状
print("Output feature shape:", feature.shape)
