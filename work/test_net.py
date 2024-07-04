import os

import torch
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
from model.network import *

path = "E:/python/data/gait/64x44/001/nm-01/"
angles = os.listdir(path)
input_data = []
_input_data = []
for angle in angles[2:9]:
    pic_path = os.path.join(path, angle)
    pic_paths = os.listdir(pic_path)
    for pic in pic_paths[0:11]:
        pic = os.path.join(pic_path, pic)
        fig = cv2.imread(pic)[:, :, 0]
        fig = torch.tensor(fig).unsqueeze(0).unsqueeze(0).float()
        img = fig.cuda()
        _input_data.append(img)
    input_data.append(torch.cat(_input_data, 1))
    _input_data = []
input_data = torch.cat(input_data, 0)



if __name__=="__main__":
    encoder = GaitCSTL_Dynamic().float()
    encoder = nn.DataParallel(encoder).cuda()
    feature, part_prob = encoder(input_data)



    print(feature, part_prob)
