import torch

from model.network import LegModel, Nolocal, LocalNet

n, t, c, h, w = 10, 5, 1, 64, 44
input_data = torch.rand(n, t, h, w).cuda()

# net = LegModel().cuda()
# net = Mymodel().cuda()
# net = Nolocal().cuda()
net = LocalNet().cuda()



feature, part_prob = net(input_data)



