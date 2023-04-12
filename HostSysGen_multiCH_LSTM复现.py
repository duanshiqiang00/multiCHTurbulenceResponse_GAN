# 多通道 紊流响应生成脉冲响应
#
import numpy
import os
from PIL import Image
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import locale
import torch.nn.functional as F
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled = False


class dataAbout():
    def load_data(dataPath,timeSignalLength):
        dataFileNameLists = os.listdir(dataPath);

        responseSignalData = numpy.empty((len(dataFileNameLists),1,3,timeSignalLength),dtype='float64');
        timedomainSysFunctionData = numpy.empty((len(dataFileNameLists),1,1,timeSignalLength),dtype='float64');

        for dataFileName in dataFileNameLists:
            dataFilePathName = dataPath+"/"+dataFileName
            allDataFile = open(dataFilePathName)
            allData = numpy.loadtxt(allDataFile)

            responseSignalData[dataFileNameLists.index(dataFileName), :, : ]= numpy.reshape(allData[:, 0:3].T,(1,3, timeSignalLength));
            timedomainSysFunctionData[dataFileNameLists.index(dataFileName), :, :] = numpy.reshape(allData[:, 3].T,(1,1, timeSignalLength));

        return responseSignalData, timedomainSysFunctionData, dataFileNameLists

    def self_train_test_split(ALlData, ALlLabel,AllFileNameList, TRAIN_TEST_RATE):
        TrainData, TestData, TrainLabel, TestLabel,trainFileName,TestFileName \
            = train_test_split(ALlData[:MAXDATASIZE, :, :], ALlLabel[:MAXDATASIZE,:,:],AllFileNameList[:MAXDATASIZE], test_size=TRAIN_TEST_RATE,shuffle=True)
        ## 此处MAXDATASIZE 表示读入 的最大数据量
        # = train_test_split(ALlData[:MAXDATASIZE, :, :, :], ALlLabel[:MAXDATASIZE], test_size=TRAIN_TEST_RATE)

        return TrainData, TestData, TrainLabel, TestLabel,trainFileName,TestFileName

    def numpyTOFloatTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.FloatTensor.float(data)
        return tensorData

    def numpyTOLongTensor(data):
        data = torch.from_numpy(data)
        tensorData = torch.LongTensor.long(data)
        return tensorData

    def modelChoice(model, data1, data2, label):

        if model == 'STFT':
            return data1, label
        if model == 'EMD':
            return data2, label
        if model == 'STFT+EMD':
            data = numpy.concatenate((data1, data2), axis=1)
            return data, label
        else:
            print("(口..口) 没有输入合适的CNN输入模式 (口..口)")
            exit()


    def NetInputLayerNum(model):
        if model == 'STFT':
            return 3
        if model == 'EMD':
            return 1
        if model == 'STFT+EMD':
            return 4
        else:
            print("(口..口) 没有输入合适的CNN输入层的数量 (口..口)")
            exit()

    # 数据封装函数
    def data_loader(data_x, data_y):

        train_data = Data.TensorDataset(data_x, data_y)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader

    def mergeTwoList(List1, List2):

        List1Size = len(List1)
        List2Size = len(List2)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(arrayList1.shape[0], arrayList1.shape[1])
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(arrayList2.shape[0], arrayList2.shape[1])
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2), axis=1)
        return mergedArrayList

    def mergeTwotimeLossList(List1, List2):

        List1Size = len(List1)
        List2Size = len(List2)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(List1Size, 1)
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(List2Size, 1)
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2), axis=1)
        return mergedArrayList

    def mergeList(List1, List2, List3):

        List1Size = len(List1)
        List2Size = len(List2)
        List3Size = len(List3)

        arrayList1 = numpy.array(List1)
        arrayList1 = arrayList1.reshape(List1Size, 1)
        arrayList2 = numpy.array(List2)
        arrayList2 = arrayList2.reshape(List2Size, 1)
        arrayList3 = numpy.array(List3)
        arrayList3 = arrayList3.reshape(List3Size, 1)
        mergedArrayList = numpy.concatenate((arrayList1, arrayList2, arrayList3), axis=1)
        return mergedArrayList

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        print('unet')
        nlayers = LayerNumber
        nefilters=NumberofFeatureChannel  ### 每次迭代时特征增加数量###
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = [3, 3]
        merge_filter_size = [3, 3]
        self.encoder = nn.ModuleList()  ### 定义一个空的modulelist命名为encoder###
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [ResCHNo] + [(i + ResCHNo) * nefilters for i in range(nlayers-1)]
        # print("echannelin",echannelin)
        echannelout = [(i + ResCHNo) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        # print("echannelout",echannelout)
        # print("dchannelout",dchannelout)
        # dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - ResCHNo) * nefilters for i in range(nlayers,1,-1)]
        # for i in range(nlayers-2, -1, -1):
        #     print('suoying',i)
        dchannelin = [dchannelout[0] * 2] + [echannelout[i+1] +echannelout[i] for i in range(nlayers-2, -1, -1)]
        # print("dchannelin",dchannelin)


        for i in range(self.num_layers):
            self.encoder.append(nn.Conv2d(echannelin[i],echannelout[i],kernel_size = filter_size,padding=filter_size[0]//2))
            self.decoder.append(nn.Conv2d(dchannelin[i],dchannelout[i],kernel_size = merge_filter_size,padding=merge_filter_size[0]//2))
            self.ebatch.append(nn.BatchNorm2d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm2d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv2d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size[0]//2), # //双斜杠取整
            nn.BatchNorm2d(echannelout[-1]),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )
        LSTMFeature = int((Fs * T) / 2 ** LayerNumber)
        # print("LSTMFeature",LSTMFeature)
        self.middleLSTM = nn.LSTM(input_size = LSTMFeature,hidden_size=LSTMFeature,batch_first=True)
        self.out = nn.Sequential(
            nn.Conv1d(dchannelout[-1], dchannelout[-1], filter_size,padding=filter_size[0]//2),
            # nn.Tanh()
            nn.LeakyReLU(0.1)
        )
        self.inputFeature1 = nn.Sequential(
            nn.Conv1d(ResCHNo, ResCHNo, kernel_size=7, padding=7 // 2),
            nn.LeakyReLU(0.1)
        )
        self.inputFeature2 = nn.Sequential(
            nn.Conv1d(ResCHNo, ResCHNo, kernel_size=3, padding=3 // 2),
            nn.LeakyReLU(0.1)
        )
        self.smooth = nn.Sequential(
            nn.Conv1d(dchannelout[-1]+ResCHNo, 1 , kernel_size=filter_size, padding=filter_size[0]//2,),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        encoder = list()
        input = x

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            print("每层编码器的输出的x的维数：",numpy.shape(x))
            encoder.append(x)
            x = x[:,:,::2]
            # print(x.shape)

        # print('middleLSTM前：',x.shape)
        x = self.middle(x)
        # h0=torch.full([1, x.shape[0], int((Fs * T) / 2 ** LayerNumber)], 0); c0=torch.full([1, x.shape[0], int((Fs * T) / 2 ** LayerNumber)], 0)
        # h0=h0.to(device);c0 =c0.to(device)
        # self.middleLSTM.flatten_parameters()
        # x,(h0, c0) = self.middleLSTM(x,(h0, c0))
        # x =torch.tensor(x)
        # print('middleLSTM后：',len(x))
        for i in range(self.num_layers):

            # x = F.interpolate(x,scale_factor=2,mode='bilinear')
            print("x_dim输出的", numpy.shape(x),
                  "encoder[self.num_layers - i - 1]的维数：", numpy.shape(encoder[self.num_layers - i - 1]))

            # print('deocder_dim：',x.shape,
            #       '\tencode_dim:',encoder[self.num_layers - i - 1].shape)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对####
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        # inputfeature1 = self.inputFeature1(input)
        # inputfeature2 = self.inputFeature2(input)
        # x = torch.cat([x, inputfeature1, inputfeature2], dim=1)

        x = self.out(x)
        x = torch.cat([x, input],dim=1)
        x = self.smooth(x)

        return x

def train_and_test(NetModel,all_data,all_label,fileNameList):
    LOSS_DATA = []
    LOSS_DATA_timeSeries=[]
    LOSS_DATA_spectrum=[]
    LOSS_TEST_DATA1 = []
    LOSS_TEST_timeSeries=[]
    LOSS_TEST_spectrun=[]
    TRAIN_ACC = []
    TEST_ACC1 = []
    TEST_ACC2 = []

    device_ids = [0, 1, 2]
    CNNNet = NetModel().to(device)
    CNNNet = nn.DataParallel(CNNNet, device_ids=device_ids)

    optimizer = torch.optim.Adam(CNNNet.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    loss_func1 = nn.SmoothL1Loss(reduction='mean')
    # loss_func = nn.CosineSimilarity()

    for epoch in range(EPOCH):

        train_data, test_data1, train_label, test_label1, train_fileName,test_fileName = \
            dataAbout.self_train_test_split(all_data, all_label, fileNameList, TRAIN_TEST_RATE)
        train_tensor_data = dataAbout.numpyTOFloatTensor(train_data)
        train_tensor_label = dataAbout.numpyTOFloatTensor(train_label)

        test_tensor_data1 = dataAbout.numpyTOFloatTensor(test_data1)
        test_tensor_label1 = dataAbout.numpyTOFloatTensor(test_label1)


        train_loadoer = dataAbout.data_loader(train_tensor_data, train_tensor_label)
        for step, (x, y) in enumerate(train_loadoer):
            # x = Variable(x)
            # y = Variable(y)
            #
            # x = x.to(device)
            # y = y.to(device)
            # # print(y.dtype)
            #
            # # with torch.no_grad():
            #
            # output = CNNNet(x)
            #
            # # print(output.shape)
            # # print(y.shape)
            #
            # '''train data time loss'''
            # timeSeriesloss = loss_func(output[:, :, :int(output.shape[2] / lossRate)],
            #                            y[:, :, :int(y.shape[2] / lossRate)])
            #
            # '''train data stft loss'''
            # ystft = torch.stft(y.view(y.shape[0], y.shape[2])[:, :int(y.shape[2] / lossRate)], Fs)
            # outputstft = torch.stft(output.view(output.shape[0], output.shape[2])[:, :int(output.shape[2] / lossRate)],
            #                         Fs)
            # spectrumLoss = loss_func(ystft, outputstft)
            # '''train data all loss'''
            # loss = 0.5 * timeSeriesloss + 0.5 * spectrumLoss
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # LOSS_DATA.append(loss.item())
            # LOSS_DATA_timeSeries.append(timeSeriesloss.item())
            # LOSS_DATA_spectrum.append(spectrumLoss.item())
            #
            # testOutput = CNNNet(test_tensor_data1)
            #
            # test_tensor_label1_GPU = test_tensor_label1.to(device)
            # lossTest1timeSeries = loss_func(testOutput[:, :, :int(testOutput.shape[2] / lossRate)],
            #                                 test_tensor_label1_GPU[:, :, int(test_tensor_label1_GPU.shape[2] / 4)])
            # '''test data time loss'''
            #
            # '''test data stft loss'''
            # testOutputSTFT = torch.stft(
            #     testOutput.view(
            #         testOutput.shape[0], testOutput.shape[2])[:, :int(testOutput.shape[2] / 4)], Fs)
            # testLabel1STFT = torch.stft(
            #     test_tensor_label1_GPU.view(
            #         test_tensor_label1_GPU.shape[0], test_tensor_label1_GPU.shape[2])[:, :int(test_tensor_label1_GPU.shape[2] / 4)],Fs)
            #
            # lossTest1Spectrum = loss_func(testLabel1STFT, testOutputSTFT)
            # '''test data all loss'''
            # lossTest1 = 0.5 * lossTest1timeSeries + 0.5 * lossTest1Spectrum
            #
            # LOSS_TEST_DATA1.append(lossTest1.item())
            # LOSS_TEST_timeSeries.append(lossTest1timeSeries.item())
            # LOSS_TEST_spectrun.append(lossTest1Spectrum.item())
            x = Variable(x)
            y = Variable(y)

            x = x.to(device)
            y = y.to(device)

            output = CNNNet(x)
            # loss_trainTime = loss_func(output, y)
            loss_trainTime = loss_func(output, y)

            ###################################################################
            ystft = torch.stft(y.view(y.shape[0], y.shape[2]), Fs)
            outputstft = torch.stft(output.view(output.shape[0], output.shape[2]), Fs)
            loss_trainSpectrum = loss_func(ystft,outputstft)
            loss = 0.5*loss_trainTime+0.5*loss_trainSpectrum
            ##################################################################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            LOSS_DATA.append(loss.item())

            testOutput = CNNNet(test_tensor_data1)

            ##########################################################################
            test_tensor_label1_GPU = test_tensor_label1.to(device)
            lossTest1timeSeries = loss_func(testOutput, test_tensor_label1_GPU)
            '''test data time loss'''

            '''test data stft loss'''
            testOutputSTFT = torch.stft(
                testOutput.view(testOutput.shape[0], testOutput.shape[2]), Fs)
            testLabel1STFT = torch.stft(
                test_tensor_label1_GPU.view(test_tensor_label1_GPU.shape[0], test_tensor_label1_GPU.shape[2]), Fs)

            lossTest1Spectrum = loss_func(testLabel1STFT, testOutputSTFT)
            '''test data all loss'''
            lossTest1 = 0.5 * lossTest1timeSeries + 0.5 * lossTest1Spectrum
            ################################################################################


            LOSS_TEST_DATA1.append(lossTest1.item())

        if epoch % 2 == 0:
            # print('hahaha', numpy.sum(test_acc == 0))
            print('Epoch: ', epoch,
                  '| train loss:  ', loss.item(), "\t",loss_trainTime.item(),'\t',loss_trainSpectrum.item(),'\t'
                  '| test1 loss: ', lossTest1.item(),'\t',lossTest1timeSeries.item(),'\t',lossTest1Spectrum.item())

        if epoch == EPOCH - 1:

            curTrainResultSaveHomePath = mkSaveModelResultdir(
                ResultSaveHomePath +
                '/Remote_ '+dataFileName+'_htGenLSTMmid_E-' + str(EPOCH) + "_LR-" + str(LR) + '_LayerN' + str(
                    LayerNumber) + '_filterN' + str(NumberofFeatureChannel))

            '''save loss'''
            resultLossDataPath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/loss_result')
            # LOSS_DATA_allMerge = dataAbout.mergeList(LOSS_DATA, LOSS_DATA_timeSeries, LOSS_DATA_spectrum)
            # LOSS_TEST_allMerge = dataAbout.mergeList(LOSS_TEST_DATA1, LOSS_TEST_timeSeries, LOSS_TEST_spectrun)
            # LOSS_mergedTrainTestData = dataAbout.mergeTwoList(LOSS_DATA_allMerge, LOSS_TEST_allMerge)

            LOSS_mergedTrainTestData = dataAbout.mergeTwotimeLossList(LOSS_DATA, LOSS_TEST_DATA1)
            plt.figure("loss")
            l1, = plt.plot(LOSS_DATA)
            l2, = plt.plot(LOSS_TEST_DATA1)
            plt.xlabel('epoch')
            plt.ylabel('loss time&spectrum')
            plt.legend(handles=[l1, l2], labels=['train loss', 'test loss'], loc='best')
            plt.title('loss time&spectrum')
            plt.savefig(
                resultLossDataPath + '/loss_time&spectrum_E-%s_LR-%f_Time-%sS.png'
                % (EPOCH, LR, T)
            )
            # plt.savefig(
            #     resultLossDataPath + '/loss_time&spectrum_E-%s_LR-%f_Time-%sS.eps'
            #     % (EPOCH, LR, T)
            # )
            plt.savefig(
                resultLossDataPath + '/loss_time&spectrum_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            plt.figure("loss time")
            l1, = plt.plot(LOSS_DATA_timeSeries)
            l2, = plt.plot(LOSS_TEST_timeSeries)
            plt.xlabel('epoch')
            plt.ylabel('loss time')
            plt.legend(handles=[l1, l2], labels=['train time loss', 'test time loss'], loc='best')
            plt.title('loss time')
            plt.savefig(
                resultLossDataPath + '/loss_time_E-%s_LR-%f_Time-%sS.png'
                % (EPOCH, LR, T)
            )
            # plt.savefig(
            #     resultLossDataPath + '/loss_time_E-%s_LR-%f_Time-%sS.eps'
            #     % (EPOCH, LR, T)
            # )
            plt.savefig(
                resultLossDataPath + '/loss_time_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            plt.figure("loss spectrum")
            l1, = plt.plot(LOSS_DATA_timeSeries)
            l2, = plt.plot(LOSS_TEST_timeSeries)
            plt.xlabel('epoch')
            plt.ylabel('loss spectrum')
            plt.legend(handles=[l1, l2], labels=['train spectrum loss', 'test spectrum loss'], loc='best')
            plt.title('loss time')
            plt.savefig(
                resultLossDataPath + '/loss_spectrum_E-%s_LR-%f_Time-%sS.png'
                % (EPOCH, LR, T)
            )
            # plt.savefig(
            #     resultLossDataPath + '/loss_spectrum_E-%s_LR-%f_Time-%sS.eps'
            #     % (EPOCH, LR, T)
            # )
            plt.savefig(
                resultLossDataPath + '/loss_spectrum_E-%s_LR-%f_Time-%sS.svg'
                % (EPOCH, LR, T)
            )
            plt.close()

            numpy.savetxt(
                resultLossDataPath + '/loss_train&Test_all&time&spectrum_E-%s_LR-%f_Time-%sS.txt'
                % (EPOCH, LR, T), LOSS_mergedTrainTestData
            )

            '''save model'''
            ResultPathModelPath = mkSaveModelResultdir(curTrainResultSaveHomePath + "/model_result")
            torch.save(
                CNNNet.state_dict(),
                ResultPathModelPath +
                '/model_state_dict.pth'
            )
            torch.save(
                CNNNet,
                ResultPathModelPath +
                '/model_NetStructure.pkl'
            )

            '''测试结果 '''
            testPredResultCompairPath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/testData_result')
            trainResultSavePath = mkSaveModelResultdir(curTrainResultSaveHomePath+'/trainData_result')
            testPredResultCompairPath = mkSaveModelResultdir(
                curTrainResultSaveHomePath + '/remoteHost_testResSys')

            for i in range(0,x.shape[0]):
                if i%2==0:
                    trainResSignal = x[i,:,:].detach().cpu().numpy().T;
                    trainTimeSys  = y[i,:,:].detach().cpu().numpy().T;
                    predTrainSys = output[i,:,:].detach().cpu().numpy().T;
                    trainMergedresult  = numpy.concatenate((trainResSignal,trainTimeSys,predTrainSys),axis=1);
                    trainSavedFileName = 'train_1Res_2Sys_3PredSys_'
                    for j in range(2, len(train_fileName[i].split('_'))):
                        trainSavedFileName += train_fileName[i].split('_')[j]
                    numpy.savetxt(
                        trainResultSavePath+'/'+trainSavedFileName,trainMergedresult)

            for i in range(0, test_tensor_data1.shape[0]):
                if i % 2 == 0:

                    testResponseSignal = test_tensor_data1[i, :, :].detach().cpu().numpy().T;
                    testTimeSys = test_tensor_label1[i, :, :].detach().cpu().numpy().T;
                    predTestTime = testOutput[i, :, :].detach().cpu().numpy().T;
                    testMergedResSysPredSys = numpy.concatenate((testResponseSignal, testTimeSys, predTestTime),
                                                                axis=1);
                    testSavedFileName = 'test_1Res_2Sys_3PredSys_'
                    for j in range(2, len(test_fileName[i].split('_'))):
                        testSavedFileName += test_fileName[i].split('_')[j]

                    numpy.savetxt(
                        testPredResultCompairPath +'/' + testSavedFileName, testMergedResSysPredSys)

def mkSaveModelResultdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path

#
# def resultShowandSave(
#         Net,lossTrainData,lossTestData1,AccTrainData,AccTestData1,modelHomePath,ResultHomePath):
#
#     constantModelHomePath = mkSaveModelResultdir(
#         modelHomePath + '/svgResult/' + CNNConstructure +  CNNInputMODEL+'_'+ResultLable+'_Time-'+str(T))
#     constantResultHomePath = mkSaveModelResultdir(
#         ResultHomePath + '/svgResult/' + CNNConstructure +  CNNInputMODEL+'_'+ResultLable+'_Time-'+str(T))
#
#     torch.save(
#         Net,
#         constantModelHomePath +
#         '/model_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.pkl'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL))
#
#     plt.figure('loss')
#     l1, = plt.plot(lossTrainData, 'r')
#     l2, = plt.plot(lossTestData1, 'b--')
#     plt.ylim(-0.01, 1.8)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(handles=[l1, l2,l3], labels=['train loss', 'test1 loss','test2 loss'], loc='best')
#     plt.title('loss')
#     plt.savefig(
#         constantResultHomePath +
#         '/loss_%s_EPOCH-%s_LR-%f_Time-%s_MDATA-%s_CNNInputMODEL-%s.svg'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='svg')
#     plt.savefig(
#         constantResultHomePath +
#         '/loss_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_NNInputMODEL-%s.eps'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='eps')
#
#     lossResultincludeTrainTest1Test2 = dataAbout.mergeList(lossTrainData,lossTestData1,lossTestData2)
#     numpy.savetxt(
#         constantResultHomePath +
#         '/ loss_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.txt'
#         % (ComparisonModel, EPOCH, LR, T, MAXDATASIZE,  CNNInputMODEL), lossResultincludeTrainTest1Test2)
#
#
#     plt.figure('accuracy')
#     l11, = plt.plot(AccTrainData, 'r')
#     l12, = plt.plot(AccTestData1, 'b--')
#     plt.ylim(0, 1.1)
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(handles=[ l11,l12,l13], labels=['train acc','test1 acc','test2 acc'], loc='best')
#     plt.title('accuracy rate')
#     plt.savefig(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MAXDATASIZE-%s_CNNInputMODEL-%s.svg'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='svg')
#     plt.savefig(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.eps'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL),dpi=600,format='eps')
#     AccResultincludeTrainTest1Test2 = dataAbout.mergeList(AccTrainData, AccTestData1, AccTestData2)
#     numpy.savetxt(
#         constantResultHomePath +
#         '/acc_%s_EPOCH-%s_LR-%f_Time-%ss_MDATA-%s_CNNInputMODEL-%s.txt'
#         % (ComparisonModel, EPOCH, LR, T,  MAXDATASIZE,  CNNInputMODEL), AccResultincludeTrainTest1Test2)
#
#     plt.show()

def main():

    response,timeSys ,fileNameList = dataAbout.load_data(path,T*Fs);
    model = Unet;
    train_and_test(model, response, timeSys,fileNameList)

if __name__=='__main__':

    dataFilepath = '/media/server/口..口/Duanshiqiang/CNNLSTMGenht_multiCH/data';
    # dataFileName = 'tur1&2&3orderRandmodel_T-20s_Fs-512'
    dataFileName =  'tur3orderRandmodel_T-20s_Fs-512'
    dataFileName = 'testDataSet'
    # dataFileName = 'tur[2-5]Multimod_F[0.1-60]_D[0.005-0.2]_Rand_20s_512'
    path = dataFilepath+'/'+dataFileName
    # /media/server/口..口/Duanshiqiang/DLGenerateFRFHwHt/data/tur3order_T-20s_Fs-512
    ResultSaveHomePath = '/media/server/口..口/Duanshiqiang/CNNLSTMGenht_multiCH/result';

    MAXDATASIZE = 5000;
    TRAIN_TEST_RATE = 0.1;

    lossRate=1

    BATCH_SIZE = int((MAXDATASIZE*(1-TRAIN_TEST_RATE))/5);
    T=20;
    Fs=512;

    LayerNumber = 5;
    NumberofFeatureChannel = 2;
    ResCHNo = 1;
    timeLength = T*Fs;

    EPOCH=50000;
    # ##kerne size =21###
    LR = 0.0001;

    main()





