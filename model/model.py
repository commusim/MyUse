import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from .network import *
from .utils import TripletSampler  # 三元组采样器


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,  # 这是啥
                 total_iter,  # 总共迭代次数
                 save_name,  # 模型保存名称
                 train_pid_num,  # 数据集划分方式
                 frame_num,  # 每个视频提取多少帧
                 model_name,  # 模型名称
                 train_source,  # 训练数据集
                 test_source,  # 测试数据集
                 img_size=64):

        self.save_name = save_name  # e.g. save_name:'GaitSet_CASIA-B_73_False_256_0.2_128_full_30'
        self.train_pid_num = train_pid_num  # 73
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim  # 256
        self.lr = lr  # 0.0001
        self.hard_or_full_trip = hard_or_full_trip  # 默认：full
        self.margin = margin  # 0.2
        self.frame_num = frame_num  # 30
        self.num_workers = num_workers  # 16
        self.batch_size = batch_size  # （8,16）
        self.model_name = model_name  # GaitSet
        self.P, self.M = batch_size  # 8,16

        self.restore_iter = restore_iter  # 0
        self.total_iter = total_iter  # 80000

        self.img_size = img_size  # 64
        '''GaitSet'''
        self.encoder = GaitSet(self.hidden_dim).float()
        # self.encoder = GaitSet_Half(self.hidden_dim).float()
        # self.encoder = GaitSet_Half_Fusion(self.hidden_dim).float()
        # self.encoder = GaitSet_HPP(self.hidden_dim).float()
        '''GaitPart'''
        # self.encoder = GaitPart().float()
        # self.encoder = GaitPart_Half().float()
        '''GaitLocal'''
        # self.encoder = GaitLocal().float()
        # self.encoder = GaitLocal_part().float()
        '''GaitSA'''
        # self.encoder = GaitSA().float()
        # self.encoder = GaitSA_prior().float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'  # 代码默认为all

    def collate_fn(self, batch):
        # batch是一个list 大小是128，每一个list有5维 (frame*64*44,数字 0-frame,角度,bg-02,id),应该是调用for in trainloder的时候才会执行这个地方，生成规定的格式
        """
        其实这个函数就是自定义DataLoader如何取样本的
        改变的也是只有data，本来data是一个样本（这个样本包含许多轮廓图），然后经过select_frame有放回的取30帧，然后再做成batch
        :param batch:[30帧张量的data，view, seq_type, label, None]都是index索引对应的
        :return:
        """
        # print(len(batch))
        batch_size = len(batch)
        """
                data = [self.__loader__(_path) for _path in self.seq_dir[index]]
                feature_num代表的是data数据所包含的集合的个数,这里一直为1，因为读取的是
                  _seq_dir = osp.join(seq_type_path, _view)
                        seqs = os.listdir(_seq_dir)  # 遍历出所有的轮廓剪影
        """
        feature_num = len(batch[0][0])
        # print(batch[0][0])
        # print(batch[0][1])
        # print(batch[0][2])
        # print(batch[0][3])
        # print(batch[0][4])
        seqs = [batch[i][0] for i in range(batch_size)]  # 对应于data
        # print(len(seqs))
        frame_sets = [batch[i][1] for i in range(batch_size)]  # 对应于 frame_set
        view = [batch[i][2] for i in range(batch_size)]  # 对应于self.view[index]
        seq_type = [batch[i][3] for i in range(batch_size)]  # 对应于self.seq_type[index]
        label = [batch[i][4] for i in range(
            batch_size)]  # 对应于self.label[index]    # 这几段代码就是从batch中分别取batch_size个对应的seqs、frame_sets、view、seq_type、label
        batch = [seqs, view, seq_type, label, None]  # batch重新组织了一下，不同于刚开始调入时候的batch格式了
        '''
                 这里的一个样本由 data, frame_set, self.view[index], self.seq_type[index], self.label[index]组成
        '''

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                # 这里的random.choices是有放回的抽取样本,k是选取次数，这里的frame_num=30
                frame_id_list = random.choices(frame_set, k=self.frame_num)  # 从所有frame数量的帧中 选取30帧，组成一个list
                _ = [feature.loc[frame_id_list].values for feature in
                     sample]  # _:(30帧,64,44)  .loc是使用标签进行索引、.iloc是使用行号进行索引
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(
            len(seqs))))  # 选取的30帧样本的ndarray与len(seqs)=128做一个键值对，然后转成一个list   # seqs：128长度的list，每个list：(30,64,44)。 map函数意为将第二个参数（一般是数组）中的每一个项，处理为第一个参数的类型。

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(
                feature_num)]  # 选取的是一个样本中的30帧，所以一个样本是一个集合，feature_num=1    # asarry和.array的作用都是转为ndarray， feature_num=1
        else:  # 全采样的话，数据就不都是30帧了，所以需要补充。batch_frames应该是只有在全采样和多个显卡的时候才会用到，否则基本用不到，先不用管
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):  # 训练
        # 加载权重
        if self.restore_iter != 0:  # 不是从0开始
            self.load(self.restore_iter)

        self.encoder.train()  # 对于有dropout和BathNorm的训练要 .train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)  # 采样器  # 采样，triplet_sampler.size：8007
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,  # 自定义从数据集中取样本的策略，但是一次只返回一个batch的indices（索引）一个batch有128个index
            collate_fn=self.collate_fn,  # 将一个list的sample组成一个mini-batch的函数
            num_workers=self.num_workers)  # 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        # 当计算机的内存充足的时候，可以设置pin_memory=True，放在内存中锁页，不放在硬盘上。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。
        train_label_set = list(self.train_source.label_set)  # 标签  # label set length:73
        train_label_set.sort()  # 对标签排序  # 里面没有005,73个id 进行排序

        _time1 = datetime.now()  # 计时
        for seq, view, seq_type, label, batch_frame in train_loader:
            # batch_frame的作用，原作者回答
            # 这个主要用于多样本的并行测试。和model中的collate_fn()呼应。测试时不同样本长度不同不能用普通方式组成batch。
            # 代码中将样本按卡的数目重新分配拼接成大的“样本”，从而实现最小空间浪费的批量测试。
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                # seq[i] = self.np2var(seq[i]).float()
                seq[i] = self.np2ts(seq[i]).float()
            if batch_frame is not None:  # 这个batch_frame是测试时用，这段白写，删了也行
                batch_frame = self.np2ts(batch_frame).int()

            """模型真正加载使用的地方"""
            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2ts(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num) = self.triplet_loss(triplet_feature,
                                                                                               triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()  # 对每个条带的loss取平均

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())  # 难样本度量损失
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())  # 全样本度量损失
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())  # loss不为0的数量
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())


            loss.backward()
            self.optimizer.step()

            if self.restore_iter % 1000 == 0:  # 打印100次迭代的训练时间
                print("100次训练时间", datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:  # 10次迭代打印
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            if self.restore_iter % 10000 == 0:
                self.save()

                # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    # def ts2var(self, x):
    #     return autograd.Variable(x).cuda()

    def np2ts(self, x):
        return torch.from_numpy(x).cuda()

    def transform(self, flag, batch_size=1):  # 测试
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2ts(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2ts(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
