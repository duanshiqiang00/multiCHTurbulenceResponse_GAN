import os
import numpy
import torch
from sklearn.model_selection import train_test_split
import  torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.fftpack import fft
import torch.nn.functional as F
from torch.autograd import Variable
import random
import datetime
import time
import numpy as np
import torch.nn as nn
import math
from scipy.signal import convolve


'''
通过multiCHAttenTurGen对仿真信号分析紊流响应，
调用Labview python节点验证attention的attention map不同频率阻尼参数对结果的影响

三通道紊流响应通过读取数据集仿真的txt得到

三通道紊流响应通过GAN Generator G计算脉冲响应信号
    ----加载模型，计算脉冲响应信号；
    ----网络输出参数可通过netG的返回变量进行调整。
imshow方式展示attention feature


'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    '''
    self-attention
    '''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k, v):
        '''
        q, k, v: shape(batch_size, n_heads, sequence_length, embedding_dimension)
        attr_mask: shape(batch_size, n_heads, sequence_length, sequence_length)
        '''

        score = torch.matmul(q.transpose(-1, -2), k) / math.sqrt(q.size(1))
        attenMap = score
        score = torch.softmax(score, dim=-1)
        return torch.matmul(score, v.transpose(-1, -2)), attenMap

class Multi_head_attention(nn.Module):
    def __init__(self, embedding_dimension, n_heads):
        super(Multi_head_attention, self).__init__()
        self.n_heads = n_heads
        filter_size = [3, 21]
        filter_size = [1, 1]
        self.embedding_dimension = embedding_dimension
        self.w_q = nn.Linear(embedding_dimension, embedding_dimension * n_heads, bias=False)
        self.w_k = nn.Linear(embedding_dimension, embedding_dimension * n_heads, bias=False)
        self.w_v = nn.Linear(embedding_dimension, embedding_dimension * n_heads, bias=False)

        self.w_qConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension * n_heads,
                filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.w_kConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension * n_heads,
                filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.w_vConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension * n_heads,
                filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.FCConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension * n_heads, embedding_dimension,
                filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.fc = nn.Linear(embedding_dimension * n_heads, embedding_dimension, bias=False)
        self.LayerNorm = nn.BatchNorm2d(self.embedding_dimension)

    def forward(self, attr_q, attr_k, attr_v):
        '''
        attr_q, attr_k, attr_v: shape(batch_size, sequence_length, embedding_dim)
        attr_mask: shape(batch_size, sequence_length, sequence_length)

        q, k, v: shape(batch_size, n_heads, sequence_length, embedding_dim)
        attr_mask expend : shape(shape(batch_size, n_heads, seq_len, seq_len)

        context : shape(batch_size, n_heads, sequence_length, embedding_dim)
        context reshape: shape(batch_size, sequence_length, n_heads*embedding_dim)
        context fc: shape(batch_size, sequence_length, embedding_dim)

                ## https://zhuanlan.zhihu.com/p/130883313
                MultiHead(Q, K, V) = Concat(head_1, ..., head-h)
                    where head_i = Attention(QW_qi, KW_ki, VW_vi)

        '''
        batch_size = attr_q.shape[0]

        attr_q = attr_q.to(device)
        attr_k = attr_k.to(device)
        attr_v = attr_v.to(device)

        q = self.w_qConv2d(attr_q)
        k = self.w_kConv2d(attr_k)
        v = self.w_vConv2d(attr_v)

        context, attenMap = Attention()(q, k, v)


        # 3.3.3 节输出为以上部分context

        context = context.transpose(-1, -2)
        # context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.embedding_dimension)
        context = self.FCConv2d(context)
        print("context.shape", context.shape)
        out = self.LayerNorm(context + attr_q)

        # return nn.LayerNorm(self.embedding_dimension)(context + attr_q) #残差+layernorm
        return out, attenMap  # 残差+layernorm


class netG(nn.Module):
    def __init__(self, LayerNumber, NumberofFeatureChannel, Fs, T, ResCHNo):
        super(netG, self, ).__init__()
        print('netG')
        nlayers = LayerNumber
        nefilters = NumberofFeatureChannel  ### 每次迭代时特征增加数量###
        self.Fs = Fs;
        self.T = T;
        self.ResCHNo = ResCHNo;
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = [3, 21]
        merge_filter_size = [3, 21]
        self.encoder = nn.ModuleList()  ### 定义一个空的modulelist命名为encoder###
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()

        echannelin = [self.ResCHNo] + [(i + self.ResCHNo) * nefilters for i in range(nlayers - 1)]
        echannelout = [(i + self.ResCHNo) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0] * 2] + [echannelout[i + 1] + echannelout[i] for i in range(nlayers - 2, -1, -1)]

        for i in range(self.num_layers):
            self.encoder.append(
                nn.Conv2d(
                    echannelin[i], echannelout[i],
                    kernel_size=filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2]))
            self.decoder.append(
                nn.Conv2d(
                    dchannelin[i], dchannelout[i],
                    kernel_size=merge_filter_size, padding=[merge_filter_size[0] // 2, merge_filter_size[1] // 2]))
            self.ebatch.append(nn.BatchNorm2d(echannelout[i]))  # moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm2d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv2d(
                echannelout[-1], echannelout[-1],
                filter_size, padding=[filter_size[0] // 2, filter_size[1] // 2]),
            nn.BatchNorm2d(echannelout[-1]),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )

        self.self_attention = Multi_head_attention(echannelout[-1], n_heads=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(
                dchannelout[-1] + ResCHNo, 1,
                kernel_size=filter_size, padding=[merge_filter_size[0] // 2, merge_filter_size[1] // 2]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv2d(dchannelout[-1] + ResCHNo, 1, filter_size, padding=[0, filter_size[1] // 2]),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        encoder = list()
        input = x

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x, 0.1)
            encoder.append(x)
            x = x[:, :, :, ::2]
            # print(x.shape)

        # x = self.middle(x)
        x, attenMap = self.self_attention(x, x, x)
        attenedFea = x
        for i in range(self.num_layers):
            x = F.interpolate(x, scale_factor=[1, 2], mode='bilinear')
            x = torch.cat([x, encoder[self.num_layers - i - 1]], dim=1)  ##特征合并过程中维数不对####
            x = self.decoder[i](x)
            lastLaDecoFea = x
            x = self.dbatch[i](x)
            x = F.leaky_relu(x, 0.1)
        x = torch.cat([x, input], dim=1)
        x = self.out(x)

        return x, attenMap, lastLaDecoFea


def turResGen(T, Fs, f1, x1, f2, x2):
    t = torch.arange(0, T, step=1 / Fs) #.unsqueeze(1)
    w1 = 2 * math.pi * f1
    w2 = 2 * math.pi * f2
    sys = numpy.exp(-x1 * w1 * t) * numpy.sin(w1 * t) + numpy.exp(-x2 * w2 * t) * numpy.sin(w2 * t)
    x = numpy.random.randn(T * Fs, 1)
    sys = sys.reshape(T*Fs, 1)
    turRes = convolve(x, sys, mode='full')
    turRes = turRes[0: T * Fs]
    return turRes


def numpyTOFloatTensor(data):
    data = torch.from_numpy(data)
    tensorData = torch.FloatTensor.float(data)
    return tensorData


def main(T, Fs, f1, x1, f2, x2, resultSaveModelAbsoluedPath, turResFilePath):
    ResCHNo = 1
    NetGLayerNo = 10
    NumberofFeatureChannel = 2
    model = netG(NetGLayerNo, NumberofFeatureChannel, Fs, T, ResCHNo).to(device)

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(resultSaveModelAbsoluedPath).items()})
    print(model)

    device_ids = [0, 1, 2]
    model = nn.DataParallel(model, device_ids=device_ids)

    turResCH1 = turResGen(T, Fs, f1, x1, f2, x2)
    turResCH1 = numpy.reshape(turResCH1, [1, 1, T * Fs])
    turResCH1 = numpyTOFloatTensor(turResCH1)
    turResCH2 = turResGen(T, Fs, f1, x1, f2, x2)
    turResCH2 = numpy.reshape(turResCH2, [1, 1, T * Fs])
    turResCH2 = numpyTOFloatTensor(turResCH2)
    turResCH3 = turResGen(T, Fs, f1, x1, f2, x2)
    turResCH3 = numpy.reshape(turResCH3, [1, 1, T * Fs])
    turResCH3 = numpyTOFloatTensor(turResCH3)
    multiCHTurSig = torch.cat((turResCH1, turResCH2, turResCH3), dim=1)
    multiCHTurSig = multiCHTurSig.view(1, 1, 3, -1)

    turResData = numpy.loadtxt(turResFilePath)
    threeCHTurRes = turResData[:, 0:3]
    realImpRes = turResData[:, 3]
    generatedImpRes = turResData[:, 4]

    print('threeCHTurRes',threeCHTurRes.shape)

    threeCHTurResTensor = numpyTOFloatTensor(threeCHTurRes)
    threeCHTurResTensor = threeCHTurResTensor.view(1, 1, 3, -1)
    threeCHTurResTensor = threeCHTurResTensor.to(device)
    genImpRes, attenMap, lastLdecFea = model(threeCHTurResTensor)

    print("attenMap.shape", attenMap.shape)

    genImpResplt = genImpRes.cpu().detach().numpy()
    genImpResplt = genImpResplt.reshape([T*Fs])
    t = numpy.arange(0, T, step = 1 / Fs)
    t = t.reshape(T * Fs)

    plt.figure()
    plt.plot(t, genImpResplt)
    plt.plot(t,generatedImpRes,'r')
    plt.show()

    print(genImpResplt)
    print('\n')
    print(generatedImpRes)
    return attenMap.cpu().double().detach().numpy()


if __name__ == '__main__':
    EPOCH = 200000
    LR = 0.00005

    Fs = 512
    T = 20

    f1 = 9.3; x1 = 0.059279
    f2 = 15.46; x2 = 0.01013015

    turResFileDir = '/media/server/口..口/Duanshiqiang/multiCHAtten_GAN/paper/仿真1'
    turResFileName = 'test_1Res_2Sys_3PredSys_RandmodelF-25.4662&D-0.073015F-9.3503&D-0.069279.txt'
    turResFilePath = turResFileDir + '/' + turResFileName

    resultSavePath = '/media/server/口..口/Duanshiqiang/multiCHAtten_GAN/result' \
                     '/Remote_ tur[2-4]Multimod_F[0.1-60]_D[0.005-0.2]_AmRand_20s_512_htGenAttmid_E-50001_LR-0.0001_LayerN10_filterN2'
    resultSaveModelPath = resultSavePath + '/model_result'
    modelName = 'model_state_dict.pth'
    str = '/media/server/口..口/Duanshiqiang/multiCHAtten_GAN/result/Remote_ tur[2-4]Multimod_F[0.1-60]_D[0.005-0.2]_AmRand_20s_512_htGenAttmid_E-50001_LR-0.0001_LayerN10_filterN2/model_result/model_state_dict.pth'
    resultSaveModelAbsoluedPath = resultSaveModelPath + '/' + modelName
    print(resultSaveModelAbsoluedPath)

    attenMap = main(T, Fs, f1, x1, f2, x2, resultSaveModelAbsoluedPath, turResFilePath)

    print('attenMap.shape', attenMap.shape)

