"""
    code by Mr.Li
    this net is used prior knowledge to improve the accuracy on the CL in CASIA-B
    we creatively just take the bottom half of the figure, with NM&BG accuracy low,
    surprised to find the CL accuracy rise some percents, so we get this net to examine
"""
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
# GaitSet
from model.network.modules.modules import HorizontalPoolingPyramid
# CSTL
from model.network.modules.cstl import MSTE, ATA
from model.network.modules.localSA import LocalBlock3D, Backbone, LocalizationST, GaussianSampleST, C3DBlock
from model.network.utils import HP, MCM, SeparateFc
from model.network.modules.self_attention import Self_attention
from model.network.modules.localSA import LocalBlock, Backbone, LocalizationST, GaussianSampleST, C3DBlock

__all__ = [
    'GaitSA',
    'GaitSA_prior',
]


class GaitSA(nn.Module):
    def __init__(self):
        super(GaitSA, self).__init__()

        in_c = [1, 32, 64, 128]
        bin_num = [8, 4, 2, 1]

        self.Backbone = Backbone()
        self.local_3d = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
                                     128, 64, 64, 128,
                                     reverse=False)
        self.fusion = nn.Sequential(
            nn.Conv3d(64 * 6, 128, kernel_size=1, bias=False),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.HP = HP(p=8)
        self.MSTE = MSTE(128, 128, 8)
        self.ATA = ATA(128, 8, 16)
        self.MCM = MCM(128, 8, 4)
        self.MCM2 = MCM(128, 8, 4)
        self.Attention = Self_attention(128, 256, 6)

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            # 取到batch_frame前_个非0数据
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        x = silho.unsqueeze(2)
        del silho
        N, S, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        feat4, feat6 = self.Backbone(x)
        """used the same feature to solve"""
        feat4_5d = feat4.view(N, S, *feat4.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        feat6_5d = feat6.view(N, S, *feat6.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()

        body = self.local_3d(feat4_5d, feat6_5d)
        weight = self.Attention(feat6_5d)
        body_weight = [body[part] * weight[:, part, :, :, :].unsqueeze(2) for part in range(len(body))]
        body_weight = torch.cat(body_weight, dim=1)
        body_weight = self.fusion(body_weight)
        body_weight = self.HP(body_weight)
        body_weight = self.MCM2(body_weight)

        N, C, S, H, W = feat6_5d.size()
        bottom = torch.split(feat6_5d, H // 2, 3)[1]
        bottom = (bottom.mean(-1) + bottom.max(-1)[0])
        bottom = self.MCM(bottom)

        all_filed = self.HP(feat6_5d)
        all_filed = all_filed.view(N, S, C, -1)
        t_f, t_s, t_l = self.MSTE(all_filed)
        new = self.ATA(t_f, t_s, t_l).permute(1, 0, 2).contiguous()

        feature = torch.cat([new, bottom, body_weight], dim=1).permute(0, 2, 1).contiguous()

        return feature, None


class GaitSA_prior(nn.Module):
    def __init__(self):
        super(GaitSA_prior, self).__init__()

        out_features = 256

        self.Backbone = Backbone()
        self.Backbone_bottom = copy.deepcopy(Backbone())
        self.Block1 = LocalBlock(LocalizationST, GaussianSampleST, C3DBlock,
                                 128, 64, 64, reverse=False)
        self.Block2 = LocalBlock(LocalizationST, GaussianSampleST, C3DBlock,
                                 128, 64, 64, reverse=False)
        self.Block3 = LocalBlock(LocalizationST, GaussianSampleST, C3DBlock,
                                 128, 64, 64, reverse=False)
        self.spatial_pool = HP(p=8)
        self.temporal_pool = MCM(128, 8, 4)

        self.spatial_pool_half = HP(p=8)
        self.temporal_pool_half = MCM(128, 8, 4)
        self.HPM = SeparateFc(16, 128, out_features)

        self.legL2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.legR2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.head2d = nn.Conv2d(64, 128, 1, 1, 0)

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, 256, 73)))

        self.fusion = nn.Sequential(
            nn.Conv3d(64 * 3 + 128, 128, kernel_size=1, bias=False),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, silho, batch_frame=None):
        if batch_frame is not None:
            # 取到batch_frame前_个非0数据
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        x = silho.unsqueeze(2)
        del silho

        N, S, C, H, W = x.size()
        x = x.view(-1, C, H, W)

        bottom = torch.split(x, H // 2, 2)[1]
        _, bottom = self.Backbone_bottom(bottom)
        bottom_5d = bottom.view(N, S, *bottom.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        bottom_5d = self.spatial_pool_half(bottom_5d)
        bottom_5d = self.temporal_pool_half(bottom_5d)

        feat4, feat6 = self.Backbone(x)

        feat4_5d = feat4.view(N, S, *feat4.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        feat6_5d = feat6.view(N, S, *feat6.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        head, legR, legL = self.Block1(feat4_5d, feat6_5d)
        # feat_half = self.spatial_pool(bottom_5d)
        gl = torch.cat([legL, legR, head, feat6_5d], dim=1)
        gl = self.fusion(gl)
        gl = self.spatial_pool(gl)
        gl = self.temporal_pool(gl)
        feature = torch.cat([bottom_5d, gl], 2)
        part_classification = feature.matmul(self.fc)

        return feature, part_classification
