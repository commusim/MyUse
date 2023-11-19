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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 2, 0).contiguous()  # [n, p, c]
        return ret


class Nolocal(nn.Module):
    def __init__(self):
        super(Nolocal, self).__init__()

        in_c = [1, 32, 64, 128]
        bin_num = [8, 4, 2, 1]
        channel = 128
        part_num = 15
        div = 1
        self.MSTE = MSTE(128, 128, part_num)
        self.ATA = ATA(128, part_num, 1)
        """3D-local extraction, bottom pipe"""
        self.Backbone = Backbone()
        self.local_3d = LocalLegBlock(LocalizationST, GaussianSampleST, C3DBlock,
                                      128, 64, 64,
                                      reverse=False)
        self.TFA = TemporalFeatureAggregator(128)

        self.legL2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.legR2d = nn.Conv2d(64, 128, 1, 1, 0)
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        self.fc = nn.Linear(23, 16)

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

        # legR, legL = self.local_3d(feat4_5d, feat6_5d)
        # N, C, S, H, W = legL.size()
        # legL = legL.permute(0, 2, 1, 3, 4).contiguous().view(N * S, C, H, -1)
        # legR = legR.permute(0, 2, 1, 3, 4).contiguous().view(N * S, C, H, -1)
        # legL = self.legL2d(legL).view(N, S, 2 * C, H, -1).permute(0, 2, 1, 3, 4).contiguous()
        # legR = self.legR2d(legR).view(N, S, 2 * C, H, -1).permute(0, 2, 1, 3, 4).contiguous()
        # legL = legL.max(-1)[0] + legL.mean(-1)
        # legR = legR.max(-1)[0] + legR.mean(-1)
        # legL = self.TFA(legL)
        # legR = self.TFA(legR)

        N, C, S, H, W = feat6_5d.size()
        bottom = torch.split(feat6_5d, H // 2, 3)[1]
        bottom = bottom.mean(-1) + bottom.max(-1)[0]
        bottom = self.TFA(bottom)

        all_filed = self.HPP(feat6)
        all_filed = all_filed.view(N, S, C, -1)
        t_f, t_s, t_l = self.MSTE(all_filed)
        new = self.ATA(t_f, t_s, t_l).permute(1, 2, 0).contiguous()

        feature = torch.cat([bottom, new], dim=2)
        feature_fc = self.fc(feature)
        # 如果需要，你还可以应用非线性变换
        feature_fc = F.relu(feature_fc)

        return feature_fc, None
