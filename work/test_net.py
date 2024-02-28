import torch

from model.network import *

n, s, c, h, w = 10, 5, 1, 64, 44
input_data = torch.rand(n, s, h, w).cuda()

hidden_dim = 256

'''GaitSet'''
# encoder = GaitSet(hidden_dim).cuda()
# encoder = GaitSet_Half(hidden_dim).cuda()
# encoder = GaitSet_Half_Fusion(hidden_dim).cuda()
# encoder = GaitSet_HPP(hidden_dim).cuda()

'''GaitPart'''
# encoder = GaitPart().cuda()
# encoder = GaitPart_Half().cuda()
'''GaitLocal'''
# encoder = GaitLocal().cuda()
# encoder = GaitLocal_part().cuda()
'''GaitSA'''
# encoder = GaitSA().cuda()
# encoder = GaitSA_prior().cuda()
encoder = GaitLocal().cuda()

feature, part_prob = encoder(input_data)

print(feature, part_prob)
