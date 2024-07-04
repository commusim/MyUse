import numpy
import torch
import torch.nn as nn


class Self_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_params=5):
        super().__init__()
        self.convs = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2 * 2 * 2, num_params),
            nn.BatchNorm1d(num_params),

        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out).unsqueeze(dim=2)
        return out.unsqueeze(dim=2).unsqueeze(dim=2)

class DymaicAttention(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,x):
        """"
        :param x:[B,S,1,H,W]
        :return: dy[B,S-1,1,H,W]
        """
        b,s,c,h,w = x.size()
        dy = torch.zeros([b,s-1,c,h,w])
        for index in range(s-1):
            dy[:,index,:] = x[:,index+1,:]-x[:,index,:]

        return dy
