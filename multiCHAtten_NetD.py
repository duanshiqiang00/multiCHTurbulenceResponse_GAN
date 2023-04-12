import numpy as np
import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math
import os
import torch.nn.functional as F

'''
mutliCHAtten_NetD 网络通过GAN的Discriminator进行生成的脉冲响应的判别损失优化进行脉冲generator的优化
inputDataSize = batchsizeX1X(T*Fs)

'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class netD(nn.Module):
    def __init__(self, NetDLayerNo, Fs, T, ResCHNo):
        super(netD, self, ).__init__()
        print('netD')
        self.Fs = Fs;
        self.T = T;
        self.ResCHNo = ResCHNo;
        self.num_layers = NetDLayerNo
        self.nefilters = 1
        filter_size = 21
        linearInFeaNo = int(self.num_layers*(self.T*self.Fs)/2**self.num_layers)
        self.encoder = nn.ModuleList()  ### 定义一个空的modulelist命名为encoder###
        self.ebatch = nn.ModuleList()

        echannelin = [self.ResCHNo] + [(i + self.ResCHNo) * self.nefilters for i in range(self.num_layers - 1)]
        echannelout = [(i + self.ResCHNo) * self.nefilters for i in range(self.num_layers)]

        for i in range(self.num_layers):
            self.encoder.append(
                nn.Conv1d(
                    echannelin[i], echannelout[i],
                    kernel_size=filter_size, padding=filter_size // 2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  # moduleList 的append对象是添加一个层到module中

        self.out = nn.Sequential(
            nn.Linear(linearInFeaNo, 5),
            # nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x, 0.1)
            x = x[:, :, ::2]
        x = x.view(x.size(0), -1) # reshaped size batchsize X ( n*((T*Fs)/2**n) )
        x = self.out(x)

        return x

if __name__ == '__main__':

    T = 10
    Fs = 512
    NetDLayerNo = 10
    ResCHNo = 1
    Batchsize = 10
    x = torch.rand([Batchsize, 1, T*Fs])
    CNNNet = netD(NetDLayerNo, Fs, T, ResCHNo)
    y = CNNNet(x)
    print(y.shape)