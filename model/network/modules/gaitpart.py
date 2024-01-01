import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              bias=False,
                              **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class FConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, p=1):
        super().__init__()
        self.conv = BasicConv2d(in_channels,
                                out_channels,
                                kernel_size,
                                padding=padding)
        self.p = p

    def forward(self, x):
        N, C, H, W = x.size()
        stripes = torch.chunk(x, self.p, dim=2)
        concated = torch.cat(stripes, dim=0)
        out = F.leaky_relu(self.conv(concated), inplace=False)
        out = torch.cat(torch.chunk(out, self.p, dim=0), dim=2)
        return out


class FPFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = FConv(1, 32, 5, 2, p=1)
        self.layer2 = FConv(32, 32, 3, 1, p=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer3 = FConv(32, 64, 3, 1, p=4)
        self.layer4 = FConv(64, 64, 3, 1, p=4)

        self.layer5 = FConv(64, 128, 3, 1, p=8)
        self.layer6 = FConv(128, 128, 3, 1, p=8)

    def forward(self, x):
        N, T, C, H, W = x.size()
        out = x.view(-1, C, H, W)

        out = self.maxpool(self.layer2(self.layer1(out)))
        out = self.maxpool(self.layer4(self.layer3(out)))
        out = self.layer6(self.layer5(out))

        _, outC, outH, outW = out.size()
        out = out.view(N, T, outC, outH, outW)
        return out


class MTB1(nn.Module):
    def __init__(self, channels=128, num_part=16, squeeze_ratio=4):
        super().__init__()

        self.avgpool = nn.AvgPool1d(3, padding=1, stride=1)
        self.maxpool = nn.MaxPool1d(3, padding=1, stride=1)

        hidden_channels = channels // squeeze_ratio
        self.conv1 = nn.Conv1d(channels * num_part,
                               hidden_channels * num_part,
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=1,
                               padding=0,
                               groups=num_part)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T = x.size()

        Sm = self.avgpool(x) + self.maxpool(x)
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        out = Sm * attention
        return out


class MTB2(nn.Module):
    def __init__(self, channels=128, num_part=16, squeeze_ratio=4):
        super().__init__()

        self.avgpool = nn.AvgPool1d(5, padding=2, stride=1)
        self.maxpool = nn.MaxPool1d(5, padding=2, stride=1)

        hidden_channels = channels // squeeze_ratio
        self.conv1 = nn.Conv1d(channels * num_part,
                               hidden_channels * num_part,
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T = x.size()

        Sm = self.avgpool(x) + self.maxpool(x)
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        out = Sm * attention
        return out


class MCM(nn.Module):
    def __init__(self, channels, num_part, squeeze_ratio=4):
        super().__init__()
        self.layer1 = MTB1(channels, num_part, squeeze_ratio)
        self.layer2 = MTB2(channels, num_part, squeeze_ratio)

    def forward(self, x):
        N, T, C, M = x.size()
        out = x.permute(0, 3, 2, 1).contiguous().view(N, M * C, T)
        out = self.layer1(out) + self.layer2(out)
        out = out.max(2)[0]
        return out.view(N, M, C)



