import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import pdb
import random

from .basic_blocks import BasicConv2d, SetBlock
from .modules.cstl import MSTE, ATA, SSFL

__all__ = [
    'GaitCSTL_Half_Fusion',
    'GaitCSTL_Half_Fusion_Easy',
    'GaitCSTL_VP',
    'GaitCSTL_Abl_SSFL',
]


class GaitCSTL_Half_Fusion(nn.Module):
    """Exclude SSFL of CSTL, using the Bottom enhance"""

    def __init__(self, hidden_dim=256, part_num=32, div=16):
        super(GaitCSTL_Half_Fusion, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]

        # 2D Convolution
        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))

        self.b_conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.b_conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.b_conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.b_conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))
        # three modules
        self.multi_scale = MSTE(_channels[2], _channels[2], part_num)
        self.adaptive_aggregation = ATA(_channels[2], part_num, div)
        # self.salient_learning = SSFL(_channels[2], _channels[2], part_num, class_num)

        # separate FC
        self.fc_bin = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(part_num, _channels[2] * 3, hidden_dim)))
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            # nn.BatchNorm3d
            nn.ReLU(inplace=True)
        )
        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho):
        n = silho.size(0)
        # silho = [n,s,h,w](chanel=1)
        x = silho.unsqueeze(2)
        # x = [n,s,1,128,88]
        del silho
        bottom = torch.split(x, x.size()[3] // 2, 3)[1]

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        bottom = self.b_conv2d_1(bottom)
        bottom = self.b_conv2d_2(bottom)
        bottom = self.b_conv2d_3(bottom)
        bottom = self.b_conv2d_4(bottom)

        split = torch.split(x, x.size()[3] // 2, 3)
        tmp = torch.cat([split[1], bottom], 2).view(-1, 256, *bottom.size()[3:])
        bottom_fusion = self.feature_fusion(tmp).view(-1, *bottom.size()[1:])
        x = torch.cat([split[0], bottom_fusion], 3)

        """HP操作"""
        x = x.max(-1)[0] + x.mean(- 1)
        # print(x.size())
        # x = [n,s,128,64] = [B,N,C,K]
        '''时间提取上进行操作，自关注特征，long中有权重，short中两层时间提取'''
        t_f, t_s, t_l = self.multi_scale(x)

        feature = self.adaptive_aggregation(t_f, t_s, t_l)

        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None


class GaitCSTL_Half_Fusion_Easy(nn.Module):
    """Exclude SSFL of CSTL, using the Bottom enhance,but the fusion is the easy pattern"""

    def __init__(self, hidden_dim=256, part_num=32, div=16):
        super(GaitCSTL_Half_Fusion_Easy, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]

        # 2D Convolution
        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))

        # three modules
        self.multi_scale = MSTE(_channels[2], _channels[2], part_num)
        self.adaptive_aggregation = ATA(_channels[2], part_num, div)
        # self.salient_learning = SSFL(_channels[2], _channels[2], part_num, class_num)

        # separate FC
        self.fc_bin = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(part_num, _channels[2] * 3, hidden_dim)))
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            # nn.BatchNorm3d
            nn.ReLU(inplace=True)
        )
        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho):
        n = silho.size(0)
        # silho = [n,s,h,w](chanel=1)
        x = silho.unsqueeze(2)
        # x = [n,s,1,128,88]
        del silho

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        bottom = torch.split(x, x.size()[3] // 2, 3)[1]
        split = torch.split(x, x.size()[3] // 2, 3)
        tmp = torch.cat([split[1], bottom], 2).view(-1, 256, *bottom.size()[3:])
        bottom_fusion = self.feature_fusion(tmp).view(-1, *bottom.size()[1:])
        x = torch.cat([split[0], bottom_fusion], 3)

        """HP操作"""
        x = x.max(-1)[0] + x.mean(- 1)
        # print(x.size())
        # x = [n,s,128,64] = [B,N,C,K]
        '''时间提取上进行操作，自关注特征，long中有权重，short中两层时间提取'''
        t_f, t_s, t_l = self.multi_scale(x)

        feature = self.adaptive_aggregation(t_f, t_s, t_l)

        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None


class GaitCSTL_VP(nn.Module):
    """HP to be replaced into VP with bottom block"""

    def __init__(self, hidden_dim=256, part_num=32, div=16):
        super(GaitCSTL_VP, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]

        # 2D Convolution
        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))
        part_num = 22
        # three modules
        self.multi_scale = MSTE(_channels[2], _channels[2], part_num)
        self.adaptive_aggregation = ATA(_channels[2], part_num, div)
        # self.salient_learning = SSFL(_channels[2], _channels[2], part_num, class_num)

        # separate FC
        self.fc_bin = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(part_num, _channels[2] * 3, hidden_dim)))
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho):
        n = silho.size(0)
        # silho = [n,s,h,w](chanel=1)
        x = silho.unsqueeze(2)
        # x = [n,s,1,128,88]
        del silho
        x = torch.split(x, x.size()[3] // 2, 3)[1]

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        """HP操作"""
        x = x.max(-2)[0] + x.mean(- 2)
        # print(x.size())
        # x = [n,s,128,64] = [B,N,C,K]
        '''时间提取上进行操作，自关注特征，long中有权重，short中两层时间提取'''
        t_f, t_s, t_l = self.multi_scale(x)

        feature = self.adaptive_aggregation(t_f, t_s, t_l)

        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None


class GaitCSTL_Abl_SSFL(nn.Module):
    """Exclude SSFL of CSTL, Ablation study"""

    def __init__(self, hidden_dim=256, part_num=32, div=16):
        super(GaitCSTL_Abl_SSFL, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]

        # 2D Convolution
        self.conv2d_1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 3, padding=1))
        self.conv2d_2 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1), True)
        self.conv2d_3 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.conv2d_4 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))

        # three modules
        self.multi_scale = MSTE(_channels[2], _channels[2], part_num)
        self.adaptive_aggregation = ATA(_channels[2], part_num, div)
        # self.salient_learning = SSFL(_channels[2], _channels[2], part_num, class_num)

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho):
        n = silho.size(0)
        # silho = [n,s,h,w](chanel=1)
        x = silho.unsqueeze(2)
        # x = [n,s,1,128,88]
        del silho

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        """HP操作"""
        x = x.max(-1)[0] + x.mean(- 1)
        # print(x.size())
        # x = [n,s,128,64] = [B,N,C,K]
        '''时间提取上进行操作，自关注特征，long中有权重，short中两层时间提取'''
        t_f, t_s, t_l = self.multi_scale(x)

        feature = self.adaptive_aggregation(t_f, t_s, t_l)

        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None
