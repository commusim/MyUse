import matplotlib.pyplot as plt
import numpy as np
import torch


def visual_sperate(x):
    """
    :param x: [B,P]
    :return:
    """
    b, s, c, h, w = x.size()
    x = x.squeeze()
    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(64, 44))
    x = x.mean(-1)
    disp = x.to('cpu').detach()
    disp = disp.numpy().T
    ax.imshow(disp,"gray")
    # left 控制左边位置；wspace，hspace 控制子图间距
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.1)
    plt.show()


def visual_feature(x, mode):
    """

    :param x:
    :param mode:
    :return:
    """
    b, s, c, h, w = x.size()
    # dynamic = torch.zeros(b, s - 1, c, h, w)
    # for index in range(s - 1):
    #     dynamic[:, index, :] = x[:, index, :]

    if (mode == "3d"):  # 创建一个图形和3D子图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 定义数据
        x = np.array(range(s))
        y = np.array(range(c))
        z = np.zeros(5)

        # 画三维柱状图
        ax.bar3d()
        # 设置轴标签
        ax.set_xlabel('Silho')
        ax.set_ylabel('')
        ax.set_zlabel('Sperate')
        plt.title('3D Bar Chart Example')
        # 显示图形
        plt.show()
    elif (mode == "2d"):
        fig, ax = plt.subplots(1, b, figsize=(64, 44))
        disp = x.mean(2).to('cpu').detach()
        disp = disp.numpy()
        for batch in range(b):
            figs = disp[batch].squeeze()
            ax[batch].imshow(figs, "gray")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.1)
        plt.show()
    return
