import torch
from model.network.modules.self_attention import Self_attention
from model.network.Dynamic import *

'''测试ImpHPP_C'''

# 定义随机输入数据
n, s, c, h, w = 2, 4, 32, 16, 11
input_data = torch.rand(n, s, c, h, w)

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
    module1 = Dynamic(c, h, w, "max")
    module2 = Regression(c, h, w, "max")

    out = module1(input_data)
    out = module2(out)
    print(out.size())
