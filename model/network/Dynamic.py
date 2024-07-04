import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Dynamic(nn.Module):
    def __init__(self, c, h, w, mode):
        super(Dynamic, self).__init__()
        self.mode = mode
        self.fc = nn.Linear(c, 1)

    def forward(self, x):
        out = None
        b, s, c, h, w = x.size()
        frames = x[:, 0:s - 1, :] - x[:, 1:s, :]
        frames = frames.permute(0, 3, 4, 2, 1).contiguous().view(b, -1, s - 1)
        match self.mode:
            case "mix":
                out = frames.mean(-1) + frames.max(-1)[0]
            case "mean":
                out = frames.mean(-1)
            case "max":
                out = frames.max(-1)[0]
        out = out.view(b, h, w, c)
        out = self.fc(out)
        out = out.squeeze()
        return out


class Dynamic_conv(nn.Module):
    def __init__(self, c, h, w, mode):
        super(Dynamic_conv, self).__init__()
        self.conv1 = nn.Conv1d(c * h * w, c * h * w, 3, 1)
        self.fc = nn.Linear(c, 1)
        self.mode = mode

    def forward(self, x):
        b, s, c, h, w = x.size()
        frames = x.permute(0, 3, 4, 2, 1).contiguous().view(b, -1, s)
        frames = self.conv1(frames)
        out = None
        match self.mode:  # frame imp
            case "mix":
                out = frames.mean(-1) + frames.max(-1)[0]
            case "mean":
                out = frames.mean(-1)
            case "max":
                out = frames.max(-1)[0]
        out = out.view(b, h, w, c)
        out = self.fc(out)  # channel imp
        out = out.squeeze()
        return out


class Regression(nn.Module):
    def __init__(self, c, h, w, mode):
        super(Regression, self).__init__()
        self.fc = nn.Linear(h * w, 1)

    def forward(self, x):
        b, h, w = x.size()
        x = x.view(b, -1)
        out = self.fc(x)

        return out
