import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from model.network.modules.local3d import Backbone
from model.network.modules.local3d import LocalBlock3D, Localization3D, MixSample, C3DBlock
from model.network.utils import HP, MCM, SeparateFc

__all__ = [
    'GaitLocal',
    'GaitLocal_part'
]


class GaitLocal(nn.Module):
    def __init__(self):
        super(GaitLocal, self).__init__()

        out_features = 256

        self.Backbone = Backbone()
        self.Block1 = LocalBlock3D(Localization3D, MixSample, C3DBlock,
                                   128, 64, 64, 128,
                                   reverse=False)
        # self.Block2 = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
        #                            128, 64, 64, 128,
        #                            reverse=False)
        # self.Block3 = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
        #                            128, 64, 64, 128,
        #                            reverse=False)
        self.spatial_pool = HP(p=16)
        self.temporal_pool = MCM(128, 16, 4)
        self.HPM = SeparateFc(16, 128, out_features)

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, 256, 73)))

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
        feat4, feat6 = self.Backbone(x)
        feat4_5d = feat4.view(N, S, *feat4.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        feat6_5d = feat6.view(N, S, *feat6.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        gl = self.Block1(feat4_5d, feat6_5d)
        feat4_5d, feat6_5d, feat4, feat6
        # gl = self.Block2(gl, gl)
        # gl = self.Block3(gl, gl)
        gl = feat6_5d
        gl = self.spatial_pool(gl)
        gl = self.temporal_pool(gl)
        # [N,M,C(128)]
        gl = self.HPM(gl)
        # [N,M,C(256)]
        # part_classification = gl.matmul(self.fc)
        return gl, None


class GaitLocal_part(nn.Module):
    def __init__(self):
        super(GaitLocal_part, self).__init__()

        out_features = 256

        self.Backbone = Backbone()
        # self.Block1 = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
        #                            128, 64, 64, 128,
        #                            reverse=False)
        # self.Block2 = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
        #                            128, 64, 64, 128,
        #                            reverse=False)
        # self.Block3 = LocalBlock3D(LocalizationST, GaussianSampleST, C3DBlock,
        #                            128, 64, 64, 128,
        #                            reverse=False)
        self.spatial_pool = HP(p=16)
        self.temporal_pool = MCM(128, 16, 4)
        self.HPM = SeparateFc(16, 128, out_features)

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, 256, 73)))

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
        feat4, feat6 = self.Backbone(x)
        feat4_5d = feat4.view(N, S, *feat4.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        feat6_5d = feat6.view(N, S, *feat6.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        # gl = self.Block1(feat4_5d, feat6_5d)
        # gl = self.Block2(gl, gl)
        # gl = self.Block3(gl, gl)
        gl = feat6_5d
        gl = self.spatial_pool(gl)
        gl = self.temporal_pool(gl)
        # [N,M,C(128)]
        gl = self.HPM(gl)
        # [N,M,C(256)]
        # part_classification = gl.matmul(self.fc)
        return gl, None
