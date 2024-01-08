import torch
from model.network.modules.self_attention import Self_attention

'''测试ImpHPP_C'''

# 定义随机输入数据
n, t, c, h, w = 13, 10, 128, 64, 44
input_data = torch.rand(n, c, t, h, w)
param_x = torch.rand(n, c, t, h, w)
#
# # 创建 ImpHPP 实例
# part_num = 64  # 选择一个合适的 part_num
# div = 1  # 选择一个合适的 div 值
# hpp = ImpHPP_C(c, part_num, div)
#
# # 调用 ImpHPP 类来计算输出
# output = hpp(input_data)
#
# print("测试通过，输出形状为:", output.size())
#
# '''测试local模块'''

if __name__ == '__main__':
    module = Self_attention(128, 256, 5)
    out = module(input_data)
    print(out.size())
