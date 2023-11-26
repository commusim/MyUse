import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
from model.network.modules.local3d import LocalBlock3D, Backbone, LocalizationST, GaussianSampleST, C3DBlock
from model.network.utils import HP, MCM, SeparateFc
from model.network.modules.modules import SetBlockWrapper, HorizontalPoolingPyramid, SeparateFCs, PackSequenceWrapper
from model.network.modules.gaitpart import TemporalFeatureAggregator
from tensorboardX import SummaryWriter


class GaitPart_HPP(nn.Module):
    def __init__(self):
        super(GaitPart_HPP, self).__init__()

        self.Backbone = Backbone()
        self.HPP = HorizontalPoolingPyramid(bin_num=[1, 2, 4, 8])
        self.TFA = TemporalFeatureAggregator(128, 4, 31)

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

        out_4, out = self.Backbone(x)
        # [n, c, s, h, w]
        out = self.HPP(out)  # [n*s, c, p]-->[n,s,c,p]-->[n,c,s,p]
        out = out.view(N, S, *out.size()[1:]).permute(0, 2, 1, 3).contiguous()
        feature = self.TFA(out)  # [n, c, p]

        return feature, None
