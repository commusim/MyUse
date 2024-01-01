import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
from model.network.modules.local3d import LocalBlock3D, Backbone, LocalizationST, GaussianSampleST, C3DBlock
from model.network.utils import HP, MCM, SeparateFc
from model.network.modules.modules import SetBlockWrapper, HorizontalPoolingPyramid, SeparateFCs, PackSequenceWrapper
from model.network.modules.gaitpart import MCM, FPFE
from tensorboardX import SummaryWriter


class GaitPart_Half(nn.Module):
    def __init__(self, out_channels=256):
        super(GaitPart_Half, self).__init__()
        self.backbone = FPFE()
        self.spatial_pool = HP(p=8)
        self.temporal_pool = MCM(channels=128, num_part=8, squeeze_ratio=4)
        self.hpm = SeparateFc(8, 128, out_channels)

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
        x = torch.split(x, H // 2, 3)[1]
        out = self.backbone(x)
        out = self.spatial_pool(out)
        out = self.temporal_pool(out)
        out = self.hpm(out)
        return out, None
