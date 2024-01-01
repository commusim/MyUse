import numpy
import torch
import torch.nn as nn


class Self_attention(nn.Module):
    def __init__(self,
                 locator,
                 sampler,
                 extractor,
                 param_channels,
                 in_channels,
                 local_channels,
                 reverse=False):
        super(Self_attention,self).__init__()

    def forward(self, x, param_x):
        
        weight = x
        return weight

class Self_Choose(nn.Module):
    def __init__(self,
                 locator,
                 sampler,
                 extractor,
                 param_channels,
                 in_channels,
                 local_channels,
                 reverse=False):
        super(Self_Choose,self).__init__()