import copy
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
# GaitSet
from model.network.modules.modules import HorizontalPoolingPyramid
# CSTL
from model.network.modules.cstl import MSTE, ATA, ImpHPP_C
from model.network.modules.local_SA import LocalLegBlock, Backbone, LocalizationST, GaussianSampleST, C3DBlock
from model.network.utils import HP, MCM, SeparateFc


class LegModel(nn.Module):
    def __init__(self):
        super(LegModel, self).__init__()

        in_c = [1, 32, 64, 128]
        bin_num = [8, 4, 2, 1]
        channel = 128
        part_num = 15
        div = 1


        self.Backbone = Backbone()
        self.local_3d = LocalLegBlock(LocalizationST, GaussianSampleST, C3DBlock,
                                      128, 64, 64,
                                      reverse=False)

        self.legL2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.legR2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.HP = HP(p=16)
        self.fc = nn.Linear(55, 16)

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

        legR, legL = self.local_3d(feat4_5d, feat6_5d)
        N, C, S, H, W = legL.size()
        legL = legL.permute(0, 2, 1, 3, 4).contiguous().view(N * S, C, H, -1)
        legR = legR.permute(0, 2, 1, 3, 4).contiguous().view(N * S, C, H, -1)
        legL = self.legL2d(legL).view(N, S, 2 * C, H, -1).permute(0, 2, 1, 3, 4).contiguous()
        legR = self.legR2d(legR).view(N, S, 2 * C, H, -1).permute(0, 2, 1, 3, 4).contiguous()
        # part_num =


        N, C, S, H, W = feat6_5d.size()
        bottom = torch.split(feat6_5d, H // 2, 3)[1]
        bottom = bottom.mean(-1) + bottom.max(-1)[0]
        bottom = self.TFA(bottom)

        all_filed = self.HPP(feat6)
        all_filed = all_filed.view(N, S, C, -1)
        t_f, t_s, t_l = self.MSTE(all_filed)
        new = self.ATA(t_f, t_s, t_l).permute(1, 2, 0).contiguous()

        feature = torch.cat([legL, legR, bottom, new], dim=2)
        feature_fc = self.fc(feature)
        # 如果需要，你还可以应用非线性变换
        feature_fc = F.relu(feature_fc)

        return feature_fc, None
