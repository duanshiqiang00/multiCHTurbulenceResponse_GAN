import numpy as np
import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math
import os
import torch.nn.functional as F

# embedding_dimension =1
# max_sequence_length = 1024
# PE = torch.zeros(max_sequence_length, embedding_dimension)
# position = torch.arange(0, max_sequence_length).unsqueeze(1)  # shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
# position = position.float()
# buff = torch.pow(1 / 10000, 2 * torch.arange(0, embedding_dimension / 2)/ embedding_dimension)  # embedding_dimension/2
# PE[:, ::2] = torch.sin(position * buff)
# PE[:, 1::2] = torch.cos(position * buff)
# plt.figure()
# plt.plot(PE.numpy())
# plt.show()

'''
input: shape(batch_size, max_sequence_length)
output: shape(batch_size, max_sequence_length, embedding_dimension)
'''

# max_sequence_length
# encode_length
# decode_length
'''上述三个参数对应最大序列长度  encoder的positional encoding的长度 decoder的position的长度'''

# def positional_encoding(x, max_sequence_length):
#     '''
#     :param x:
#     :param max_sequence_length:
#     :return:
#     位置编码的格式为
#     PE(pos, 2i) = sin(pos/10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#     d_model = embedding_dimension =1
#     PE的维度为【seq最大长度， 编码维数】
#     '''
#     PE = torch.zeros(max_sequence_length, embedding_dimension)
#     position = torch.arange(0, max_sequence_length).unsqueeze(1) #shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
#     position = position.float()
#     buff = torch.pow(1 / 10000, 2*torch.arange(0, embedding_dimension/2)/embedding_dimension)  # embedding_dimension/2
#     PE[:, ::2] = torch.sin(position * buff)
#     PE[:, 1::2] = torch.cos(position * buff)
#     return PE
#矩阵乘积也就是不带bias的nn.Linear
# nn.Linear(embedding_dimension, embedding_dimension, bias=False)

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

        score = torch.matmul(q.transpose(-1, -2), k)  / math.sqrt(q.size(1))

        # k为四维张量，不能用转置 k.transpose(-1, -2)
        score = torch.softmax(score, dim=-1)
        return torch.matmul(score, v.transpose(-1, -2))

    def padding_mask(k, q):
        '''
        用于attention计算softmax前，对pad位置进行mask，使得softmax时该位置=0
        k, q: shape(batch_size, sequence_lenth)
        '''
        batch_size, seq_k = k.size()
        batch_size, seq_q = q.size()
        # P = 0
        mask = k.data.eq(0).unsqueeze(1)  # shape(batch_size, 1, sequence_length)
        return mask.expand(batch_size, seq_k, seq_q)

class Multi_head_attention(nn.Module):
    def __init__(self, embedding_dimension,n_heads):
        super(Multi_head_attention, self).__init__()
        self.n_heads = n_heads
        filter_size = [3, 21]
        # filter_size = [1, 1]
        self.embedding_dimension = embedding_dimension
        self.w_q = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)
        self.w_k = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)
        self.w_v = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)

        self.w_qConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension*n_heads,
                filter_size, padding=[filter_size[0]//2, filter_size[1]//2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.w_kConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension * n_heads,
                filter_size, padding=[filter_size[0]//2, filter_size[1]//2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.w_vConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension, embedding_dimension * n_heads,
                filter_size, padding=[filter_size[0]//2, filter_size[1]//2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.FCConv2d = nn.Sequential(
            nn.Conv2d(
                embedding_dimension * n_heads,embedding_dimension,
                filter_size, padding=[filter_size[0]//2, filter_size[1]//2], bias=False),
            # nn.LeakyReLU(0.1)
        )
        self.fc = nn.Linear(embedding_dimension*n_heads, embedding_dimension, bias=False)
        self.LayerNorm = nn.BatchNorm2d(self.embedding_dimension)

        '''
        Epoch:  50004 | train loss:	 0.19881226122379303 	 0.0021564781200140715 	 0.3954680562019348 	 | MSE time loss:	 0.0007769775693304837 	| test1 loss:	 0.17527592182159424 	 0.0016971769509837031 	 0.348854660987854
        Epoch:  50006 | train loss:	 0.18124650418758392 	 0.0017071462934836745 	 0.3607858717441559 	 | MSE time loss:	 0.0007008754182606936 	| test1 loss:	 0.1688222736120224 	 0.00180146680213511 	 0.3358430862426758
        Epoch:  50008 | train loss:	 0.17022256553173065 	 0.0019238356035202742 	 0.3385213017463684 	 | MSE time loss:	 0.0006426565232686698 	| test1 loss:	 0.17736220359802246 	 0.0018078901339322329 	 0.35291650891304016
        Epoch:  50010 | train loss:	 0.19857171177864075 	 0.0023320764303207397 	 0.39481133222579956 	 | MSE time loss:	 0.0007923324592411518 	| test1 loss:	 0.17586028575897217 	 0.0017250103410333395 	 0.34999555349349976
        Epoch:  50012 | train loss:	 0.18396931886672974 	 0.0020987428724765778 	 0.3658398985862732 	 | MSE time loss:	 0.0006911540403962135 	| test1 loss:	 0.20212005078792572 	 0.0022173484321683645 	 0.40202274918556213
        Epoch:  50014 | train loss:	 0.19985902309417725 	 0.0021473318338394165 	 0.39757072925567627 	 | MSE time loss:	 0.0007647956954315305 	| test1 loss:	 0.18607020378112793 	 0.0016687740571796894 	 0.37047162652015686
        
        '''


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

        # q = q.view(attr_q.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)
        # k = k.view(attr_k.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)
        # v = v.view(attr_v.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)

        context = Attention()(q, k, v)
        #3.3.3 节输出为以上部分context

        context = context.transpose(-1, -2)
        # context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.embedding_dimension)
        context = self.FCConv2d(context)
        out = self.LayerNorm(context + attr_q)

        # return nn.LayerNorm(self.embedding_dimension)(context + attr_q) #残差+layernorm
        return out #残差+layernorm


class netG(nn.Module):
    def __init__(self, LayerNumber, NumberofFeatureChannel, Fs, T, ResCHNo):
        super(netG,self,).__init__()
        print('netG')
        nlayers = LayerNumber
        nefilters=NumberofFeatureChannel  ### 每次迭代时特征增加数量###
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

        echannelin = [self.ResCHNo] + [(i + self.ResCHNo) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + self.ResCHNo) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0] * 2] + [echannelout[i+1] +echannelout[i] for i in range(nlayers-2, -1, -1)]

        for i in range(self.num_layers):
            self.encoder.append(
                nn.Conv2d(
                    echannelin[i],echannelout[i],
                    kernel_size=filter_size, padding=[filter_size[0]//2, filter_size[1]//2]))
            self.decoder.append(
                nn.Conv2d(
                    dchannelin[i],dchannelout[i],
                    kernel_size=merge_filter_size, padding=[merge_filter_size[0]//2, merge_filter_size[1]//2]))
            self.ebatch.append(nn.BatchNorm2d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm2d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv2d(
                echannelout[-1], echannelout[-1],
                filter_size, padding=[filter_size[0]//2,filter_size[1]//2]),
            nn.BatchNorm2d(echannelout[-1]),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )

        timeFeature_Len = int ((self.T*self.Fs)/(self.nefilters**self.num_layers))  ## time dimension feature 更像是seq length
        # self.self_attention = Multi_head_attention(timeFeature_Len, n_heads=6)
        self.self_attention = Multi_head_attention(echannelout[-1], n_heads=1)


        # LSTMFeature = int((Fs * T) / 2 ** LayerNumber)
        # # print("LSTMFeature",LSTMFeature)
        # self.middleLSTM = nn.LSTM(input_size = LSTMFeature,hidden_size=LSTMFeature,batch_first=True)
        # self.out = nn.Sequential(
        #     nn.Conv1d(dchannelout[-1], dchannelout[-1], filter_size,padding=filter_size[0]//2),
        #     # nn.Tanh()
        #     nn.LeakyReLU(0.1)
        # )
        # self.inputFeature1 = nn.Sequential(
        #     nn.Conv1d(ResCHNo, ResCHNo, kernel_size=7, padding=7 // 2),
        #     nn.LeakyReLU(0.1)
        # )
        # self.inputFeature2 = nn.Sequential(
        #     nn.Conv1d(ResCHNo, ResCHNo, kernel_size=3, padding=3 // 2),
        #     nn.LeakyReLU(0.1)
        # )
        self.smooth = nn.Sequential(
            nn.Conv2d(
                dchannelout[-1]+ResCHNo, 1,
                kernel_size=filter_size, padding=[merge_filter_size[0]//2, merge_filter_size[1]//2]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv2d(dchannelout[-1]+ResCHNo, 1, filter_size,padding=[0,filter_size[1]//2]),
            # nn.Tanh(),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        encoder = list()
        input = x

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:, :, :, ::2]
            # print(x.shape)

        # x = self.middle(x)
        x = self.self_attention(x, x, x)
        attenFeaMap = x;

        for i in range(self.num_layers):
            x = F.interpolate(x, scale_factor=[1, 2], mode='bilinear')

            # x = F.interpolate(x,scale_factor=2,mode='bilinear')
            # print('deocder_dim：',x.shape,
            #       '\tencode_dim:',encoder[self.num_layers - i - 1].shape)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对####
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        # inputfeature1 = self.inputFeature1(input)
        # inputfeature2 = self.inputFeature2(input)
        # x = torch.cat([x, inputfeature1, inputfeature2], dim=1)

        # x = self.out(x)
        x = torch.cat([x, input],dim=1)
        x = self.out(x)

        return x, attenFeaMap


# class Unet(nn.Module):
#     def __init__(self, LayerNumber, NumberofFeatureChannel, Fs, T):
#         super(Unet,self).__init__()
#         # print('unet')
#         # nlayers = LayerNumber
#         # nefilters=NumberofFeatureChannel  ### 每次迭代时特征增加数量###
#         self.num_layers = LayerNumber
#         self.nefilters = NumberofFeatureChannel
#         self.Fs = Fs
#         self.T = T
#         filter_size = 15
#         merge_filter_size = 15
#         self.encoder = nn.ModuleList()  ### 定义一个空的modulelist命名为encoder###
#         self.decoder = nn.ModuleList()
#         self.ebatch = nn.ModuleList()
#         self.dbatch = nn.ModuleList()
#         echannelin = [1] + [(i + 1) * self.nefilters for i in range(self.num_layers-1)]
#         echannelout = [(i + 1) * self.nefilters for i in range(self.num_layers)]
#         dchannelout = echannelout[::-1]
#         dchannelin = [dchannelout[0]*2]+[(i) * self.nefilters + (i - 1) * self.nefilters for i in range(self.num_layers,1,-1)]
#
#         for i in range(self.num_layers):
#             self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
#             self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
#             self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
#             self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
#
#         self.middle = nn.Sequential(
#             nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2), # //双斜杠取整
#             nn.BatchNorm1d(echannelout[-1]),
#             # nn.LeakyReLU(0.1)
#             nn.Tanh()
#         )
#
#         #################################################################################################
#         ## attention 的ht generation
#         seq_len = echannelout[-1]; ## encoder output feature dimension
#         embedding_dimension = int ((self.T*self.Fs)/(self.nefilters**self.num_layers))  ## time dimension feature
#
#         convFeatureDim = echannelout[-1] ## 更像是 embedding dimension
#         timeFeature_Len = int ((self.T*self.Fs)/(self.nefilters**self.num_layers))  ## time dimension feature 更像是seq length
#
#         self.self_attention = Multi_head_attention(timeFeature_Len, n_heads=6)
#         ##################################################################################################
#
#         #################################################################################################
#         ## LSTM 的ht  generation
#         LSTMFeature = int((Fs * T) / 2 ** LayerNumber)
#         self.middleLSTM = nn.LSTM(input_size = LSTMFeature,hidden_size=LSTMFeature,batch_first=True)
#
#         ######################################################################################################
#
#         self.out = nn.Sequential(
#             nn.Conv1d(self.nefilters+2, self.nefilters+2, filter_size,padding=filter_size//2),
#             # nn.Tanh()
#             nn.LeakyReLU(0.1)
#         )
#         self.inputFeature1 = nn.Sequential(
#             nn.Conv1d(1, 1, kernel_size=7, padding=7 // 2),
#             nn.LeakyReLU(0.1)
#         )
#         self.inputFeature2 = nn.Sequential(
#             nn.Conv1d(1, 1, kernel_size=3, padding=3 // 2),
#             nn.LeakyReLU(0.1)
#         )
#
#         self.smooth = nn.Sequential(
#             nn.Conv1d(self.nefilters+1, 1, kernel_size=filter_size, padding=filter_size//2,),
#             nn.LeakyReLU(0.1)
#         )
#     def forward(self,x):
#         encoder = list()
#         input = x     #x.shape[batch_size, 1-channel, 10240-T*Fs-->feature]
#
#         for i in range(self.num_layers):
#             print('+++++++++',i)
#             x = self.encoder[i](x)
#             x = self.ebatch[i](x)
#             x = F.leaky_relu(x,0.1)
#             encoder.append(x)
#             x = x[:,:,::2]
#
#         ##############################################################
#         ## attention 中间层的ht generation
#
#         #last layer encoder size[batch_size, last layer encoder out size(echannelout[-1]), featureNum( T*Fs/(self.nefilters^self.num_layers) )]
#
#         # x = x.transpose(2, 1)
#         x = self.self_attention(x, x, x)
#         # x= x.transpose(2,1)
#
#
#         ########################################################
#
#         ########################################################
#         ## LSMT 中间层的ht generation
#
#         # h0=torch.full(
#         #     [1, x.shape[0], int((self.Fs * self.T) / 2 ** self.num_layers)], 0);
#         # c0=torch.full([1, x.shape[0], int((self.Fs * self.T) / 2 ** self.num_layers)], 0)
#         # h0=h0.to(device);c0 =c0.to(device)
#         # self.middleLSTM.flatten_parameters()
#         # x,(h0, c0) = self.middleLSTM(x,(h0, c0))
#
#         #######################################################################
#
#
#         for i in range(self.num_layers):
#             # x = F.upsample(x,scale_factor=2,mode='linear')
#             x = F.interpolate(x, scale_factor = 2, mode='linear')
#             x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对####
#             x = self.decoder[i](x)
#             x = self.dbatch[i](x)
#             x = F.leaky_relu(x,0.1)
#
#         # inputfeature1 = self.inputFeature1(input)
#         # inputfeature2 = self.inputFeature2(input)
#         # x = torch.cat([x, inputfeature1, inputfeature2], dim=1)
#         # x = self.out(x)
#
#         x = torch.cat([x, input],dim=1)
#         x = self.smooth(x)
#
#         return x