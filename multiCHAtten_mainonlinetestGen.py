from Host.multiCHAtten_NetG import netG
from Host.multiCHAtten_NetD import netD
from Host.timeLoss import TimeDomainLoss_v1
from Host.stftLoss import MultiResolutionSTFTLoss
import numpy
import os
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F


'''
GAN模型训练 

读取数据
    ----dataAbout类获取数据
    ----Host.multiCHAtten_NetG and Host.multiCHAtten_NetD构成GAN的生成器和判别器
    ----train_and_test训练
    
保存结果
    ----保存仿真和实测数据（主要验证服务器和本地model load计算不一致的问题）
    
'''




os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.backends.cudnn.enabled = False

class dataAbout():
    def load_data(dataPath, timeSignalLength):
        dataFileNameLists = os.listdir(dataPath);

        responseSignalData = numpy.empty((len(dataFileNameLists), 1, 3, timeSignalLength), dtype='float64');
        timedomainSysFunctionData = numpy.empty((len(dataFileNameLists), 1, 1, timeSignalLength), dtype='float64');

        for dataFileName in dataFileNameLists:
            dataFilePathName = dataPath + "/" + dataFileName
            allDataFile = open(dataFilePathName)
            allData = numpy.loadtxt(allDataFile)

            # print('allData.shape',allData.shape)
            responseSignalData[dataFileNameLists.index(dataFileName), :, :] = \
                numpy.reshape(allData[0:timeSignalLength, 0:3].T, (1, 3, timeSignalLength));
            timedomainSysFunctionData[dataFileNameLists.index(dataFileName), :, :] = \
                numpy.reshape(allData[0:timeSignalLength, 3].T, (1, 1, timeSignalLength));

        return responseSignalData, timedomainSysFunctionData, dataFileNameLists

    def self_train_test_split(ALlData, ALlLabel, AllFileNameList, TRAIN_TEST_RATE):
        TrainData, TestData, TrainLabel, TestLabel, trainFileName, TestFileName \
            = train_test_split(ALlData[:MAXDATASIZE, :, :], ALlLabel[:MAXDATASIZE, :, :], AllFileNameList[:MAXDATASIZE],
                               test_size=TRAIN_TEST_RATE, shuffle=True)
        ## 此处MAXDATASIZE 表示读入 的最大数据量
        # = train_test_split(ALlData[:MAXDATASIZE, :, :, :], ALlLabel[:MAXDATASIZE], test_size=TRAIN_TEST_RATE)

        return TrainData, TestData, TrainLabel, TestLabel, trainFileName, TestFileName

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


def train_and_test(NetModelG, NetModelD, all_data, all_label, fileNameList):
    LOSS_DATA = []
    LOSS_TEST_DATA1 = []

    device_ids = [0, 1, 2]
    NetG = NetModelG(NetGLayerNo, NumberofFeatureChannel, Fs, T, ResCHNo).to(device)
    NetG = nn.DataParallel(NetG, device_ids=device_ids)

    optimizerG = torch.optim.Adam(NetG.parameters(), lr=LR_G)
    loss_func = nn.MSELoss()

    NetD = NetModelD(NetDLayerNo, Fs, T, ResCHNo).to(device)
    NetD = nn.DataParallel(NetD, device_ids=device_ids)
    optimizerD = torch.optim.Adam(NetD.parameters(), lr=LR_D)
    loss_D_func = nn.MSELoss()
    real_label = 1
    fake_label = 0

    for epoch in range(EPOCH):

        train_data, test_data1, train_label, test_label1, train_fileName, test_fileName = \
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

            net_D_label = torch.FloatTensor(y.size(0));

            ##########################################################################################
            '''Generator '''
            optimizerG.zero_grad()
            output, _ = NetG(x)

            '''three part self define loss '''
            '''time loss including eneragy time phase'''
            loss_trainTime_t = loss_func(output, y)
            loss_trainTime_e = loss_func(output ** 2, y ** 2)
            y_phase = F.pad(y.transpose(1, 2), (1, 0), "constant", 0) - F.pad(y.transpose(1, 2), (0, 1),
                                                                              "constant", 0)
            y_hat_phase = F.pad(output.transpose(1, 2), (1, 0), "constant", 0) - F.pad(output.transpose(1, 2),
                                                                                       (0, 1), "constant", 0)
            loss_trainTime_p = loss_func(y_phase, y_hat_phase)

            loss_trainTime = loss_trainTime_e + loss_trainTime_t + loss_trainTime_p

            '''spectrum loss      Spectral convergence and Log STFT magnitude loss'''
            ystft = torch.stft(y.view(y.size(0), -1), Fs)
            outputstft = torch.stft(output.view(output.size(0), -1), Fs)

            sc_train_loss = torch.norm(ystft - outputstft, p="fro") / torch.norm(ystft,
                                                                                 p="fro")  # Spectral convergence loss
            ''' log stft 幅度损失存在重大问题  计算之后损失不降'''
            mag_train_loss = loss_func(ystft, outputstft)  # Log STFT magnitude loss

            loss_trainSpectrum = sc_train_loss + mag_train_loss

            net_D_fake_output = NetD(output.view(output.size(0), -1, T * Fs))
            lossD_dis = loss_D_func(net_D_fake_output, torch.ones_like(net_D_fake_output))

            # loss = 1 * loss_trainTime + 1 * loss_trainSpectrum  #+ 0.1 * lossD_dis
            if loss_trainSpectrum > 1.2:
                loss = 1 * loss_trainTime + 1 * loss_trainSpectrum + 0.5 * lossD_dis
            elif loss_trainSpectrum < 1.2 and loss_trainSpectrum > 0.6:
                loss = 1 * loss_trainTime + 1 * loss_trainSpectrum + 0.35 * lossD_dis
            else:
                loss = 1 * loss_trainTime + 1 * loss_trainSpectrum + 0.2 * lossD_dis

            loss.backward()
            optimizerG.step()
            '''END  Generator'''
            ############################################################################################################

            ############################################################################################################
            '''Discriminator'''
            optimizerD.zero_grad()
            net_D_real_output = NetD(y.view(y.size(0), -1, T * Fs))
            lossD_real = loss_D_func(net_D_real_output, torch.ones_like(net_D_real_output))

            genedImpluseRes, _ = NetG(x)
            net_D_fake_output = NetD(genedImpluseRes.view(genedImpluseRes.size(0), -1, T * Fs))
            lossD_fake = loss_D_func(net_D_fake_output, torch.zeros_like(net_D_fake_output))
            lossD = lossD_real + lossD_fake

            lossD.backward()
            optimizerD.step()
            '''END Discriminator '''
            ####################################################################################

            LOSS_DATA.append(loss.item())

            testOutput, testAttenFeaMap = NetG(test_tensor_data1)

            ''' self define loss TEST '''
            test_tensor_label1_GPU = test_tensor_label1.to(device)
            loss_testTime_t = loss_func(testOutput, test_tensor_label1_GPU)
            loss_testTime_e = loss_func(testOutput ** 2, test_tensor_label1_GPU ** 2)
            y_test_phase = F.pad(test_tensor_label1_GPU.transpose(1, 2), (1, 0), "constant", 0) - F.pad(
                test_tensor_label1_GPU.transpose(1, 2), (0, 1),
                "constant", 0)
            y_test_hat_phase = F.pad(testOutput.transpose(1, 2), (1, 0), "constant", 0) - F.pad(
                testOutput.transpose(1, 2),
                (0, 1), "constant", 0)
            loss_testTime_p = loss_func(y_test_phase, y_test_hat_phase)
            loss_testTime_total = loss_testTime_e + loss_testTime_t + loss_testTime_p

            '''test data time loss'''

            '''test data stft loss'''
            testOutputSTFT = torch.stft(
                testOutput.view(testOutput.size(0), -1), Fs)
            testLabelSTFT = torch.stft(
                test_tensor_label1_GPU.view(test_tensor_label1_GPU.size(0), -1), Fs)

            sc_test_loss = torch.norm(testLabelSTFT - testOutputSTFT, p="fro") / torch.norm(ystft,
                                                                                            p="fro")  ##Spectral convergence loss
            mag_test_loss = loss_func(testLabelSTFT, testOutputSTFT)  # Log STFT magnitude loss 不能log!!!!
            loss_testSpectrum = sc_test_loss + mag_test_loss
            # loss_testSpectrum = loss_func(testLabelSTFT, testOutputSTFT)
            '''test data all loss'''
            loss_test = 0.5 * loss_testTime_total + 0.5 * loss_testSpectrum
            ################################################################################

            LOSS_TEST_DATA1.append(loss_test.item())

        if epoch % 2 == 0:
            # print('hahaha', numpy.sum(test_acc == 0))
            loss_trainTime_t = loss_func(output, y)
            print('Epoch: ', epoch,
                  '| train loss:\t', loss.item(), '\t', loss_trainTime.item(), '\t', loss_trainSpectrum.item(), '\t',
                  '| MSE time loss:\t', loss_trainTime_t.item(), '\t',
                  '| lossD_real:\t', lossD_real.item(), '\tlossD_fake:\t', lossD_fake.item(), '\tlossD_dis:\t',
                  lossD_dis.item(),

                  # '| spec train loss:\t',lossSpec.item(),'\t',loss_trainTimeSpec.item(),'\t',loss_trainSpectrumSpec.item(), '\t',
                  '| test1 loss:\t', loss_test.item(), '\t', loss_testTime_total.item(), '\t', loss_testSpectrum.item())

        if epoch == EPOCH - 1:

            curTrainResultSaveHomePath = mkSaveModelResultdir(
                ResultSaveHomePath +
                '/Remote_ ' + dataFileName + '_htGenAttmid_E-' + str(EPOCH) + "_LRG-" + str(LR_G) + '_LayerN' + str(
                    NetGLayerNo) + '_filterN' + str(NumberofFeatureChannel))

            '''save loss'''
            resultLossDataPath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/loss_result')

            LOSS_mergedTrainTestData = dataAbout.mergeTwotimeLossList(LOSS_DATA, LOSS_TEST_DATA1)
            plt.figure("loss")
            l1, = plt.plot(LOSS_DATA)
            l2, = plt.plot(LOSS_TEST_DATA1)
            plt.xlabel('epoch')
            plt.ylabel('loss time&spectrum')
            plt.legend(handles=[l1, l2], labels=['train loss', 'test loss'], loc='best')
            plt.title('loss time&spectrum')
            plt.savefig(
                resultLossDataPath + '/loss_time&spectrum_E-%s_LRG-%f_Time-%sS.png'
                % (EPOCH, LR_G, T)
            )
            plt.close()

            numpy.savetxt(
                resultLossDataPath + '/loss_train&Test_all&time&spectrum_E-%s_LR-%f_Time-%sS.txt'
                % (EPOCH, LR_G, T), LOSS_mergedTrainTestData
            )

            '''save model'''
            ResultPathModelPath = mkSaveModelResultdir(curTrainResultSaveHomePath + "/model_result")
            torch.save(
                NetG.state_dict(),
                ResultPathModelPath +
                '/model_state_dict.pth'
            )
            torch.save(
                NetG,
                ResultPathModelPath +
                '/model_NetStructure.pkl'
            )

            '''测试结果 '''
            testPredResultCompairPath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/testData_result')
            trainResultSavePath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/trainData_result')

            for i in range(0, x.shape[0]):
                if i % 2 == 0:
                    trainResSignal = x[i, :, :].detach().cpu().numpy().T;
                    trainTimeSys = y[i, :, :].detach().cpu().numpy().T;
                    predTrainSys = output[i, :, :].detach().cpu().numpy().T;
                    trainMergedresult = numpy.concatenate((trainResSignal, trainTimeSys, predTrainSys), axis=1);
                    trainSavedFileName = 'train_1Res_2Sys_3PredSys_'
                    for j in range(2, len(train_fileName[i].split('_'))):
                        trainSavedFileName += train_fileName[i].split('_')[j]

                    numpy.savetxt(
                        trainResultSavePath + '/' + trainSavedFileName,
                        trainMergedresult.reshape(trainMergedresult.shape[0], trainMergedresult.shape[1]))

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
                        testPredResultCompairPath + '/' + testSavedFileName,
                        testMergedResSysPredSys.reshape(testMergedResSysPredSys.shape[0],
                                                        testMergedResSysPredSys.shape[1]))

                    # attention feature map 保存
                    testAttenFeaMapFileName = testSavedFileName[:len(testSavedFileName) - 4] + '_AttenFeaMap.txt'
                    testAttenFeaMapFilePath = testPredResultCompairPath + "/" + testAttenFeaMapFileName

                    TestDataAttenFeaMapNumDim = testAttenFeaMap.size() #batchsizeXchannelX3X10
                    TestDataAttenFeaMapNum = \
                        numpy.zeros(shape=[TestDataAttenFeaMapNumDim[2], TestDataAttenFeaMapNumDim[3]])
                    testAttenFeaMapFile = open(testAttenFeaMapFilePath, 'wb')
                    for CH_index in range(TestDataAttenFeaMapNumDim[1]):
                        testDataAttenFeaMappir = testAttenFeaMap[i, CH_index, :, :]
                        testDataAttenFeaMappirNumpy = testDataAttenFeaMappir.cpu().detach().numpy()
                        TestDataAttenFeaMapNum = \
                            numpy.concatenate((TestDataAttenFeaMapNum, testDataAttenFeaMappirNumpy),
                                              axis=0)
                    TestDataAttenFeaMapNum = TestDataAttenFeaMapNum[3:, :]
                    numpy.savetxt(testAttenFeaMapFile, TestDataAttenFeaMapNum)
                testAttenFeaMapFile.close()

            '''测试结束'''
            ############################################################################################################

            #
            # '''颤振试飞试验实测数据验证'''
            # flutterTestDataFileResultCompairPath = mkSaveModelResultdir(curTrainResultSaveHomePath + '/' + flutterTestDataFileName)
            # realDataDirPath = dataFilepath + '/' + flutterTestDataFileName
            # fileNameList = os.listdir(realDataDirPath)
            # print('fileNameList', fileNameList)
            #
            # for fileName in fileNameList:
            #     realDataFileName = fileName
            #     realDataFilePath = realDataDirPath + '/' + realDataFileName
            #
            #     file = open(realDataFilePath, 'r')
            #     data = numpy.loadtxt(file)
            #     multiCHTurSigNumpy = data
            #     data = data[:20 * Fs]
            #
            #     signal = numpy.reshape(data, [1, 1, 3, -1])
            #     # signal = (signal - numpy.mean(signal)) / numpy.std(signal)
            #     multiCHTurSig = dataAbout.numpyTOFloatTensor(signal)
            #     multiCHTurSig = multiCHTurSig.to(device)
            #     genImpRes, flutterTestDataAttenFeaMap = NetG(multiCHTurSig)
            #
            #     MultiCHTurResFilePath = flutterTestDataFileResultCompairPath + "/" + realDataFileName
            #     numpy.savetxt(MultiCHTurResFilePath, multiCHTurSigNumpy)
            #
            #     predhtFileName = realDataFileName[:len(realDataFileName) - 4] + '-predht.txt'
            #     predhtFilePath = flutterTestDataFileResultCompairPath + "/" + predhtFileName
            #     numpy.savetxt(predhtFilePath, genImpRes.cpu().detach().numpy().flatten())
            #
            #     # attention feature map 保存
            #     predhtAttenFeaMapFileName = realDataFileName[:len(realDataFileName) - 4] + '-AttenFeaMap.txt'
            #     predhtAttenFeaMapFilePath = flutterTestDataFileResultCompairPath + "/" + predhtAttenFeaMapFileName
            #
            #     flutterTestDataAttenFeaMapNumDim = flutterTestDataAttenFeaMap.size()
            #
            #     flutterTestDataAttenFeaMapNum = \
            #         numpy.zeros(shape=[flutterTestDataAttenFeaMapNumDim[2], flutterTestDataAttenFeaMapNumDim[3]])
            #     predhtAttenFeaMapFile = open(predhtAttenFeaMapFilePath, 'wb')
            #     for i in range(flutterTestDataAttenFeaMap.shape[1]):
            #         flutterTestDataAttenFeaMappir = flutterTestDataAttenFeaMap[0, i, :, :]
            #         flutterTestDataAttenFeaMappirNumpy = flutterTestDataAttenFeaMappir.cpu().detach().numpy()
            #         flutterTestDataAttenFeaMapNum = \
            #             numpy.concatenate((flutterTestDataAttenFeaMapNum, flutterTestDataAttenFeaMappirNumpy), axis=0)
            #     flutterTestDataAttenFeaMapNum = flutterTestDataAttenFeaMapNum[3:, :]
            #     numpy.savetxt(predhtAttenFeaMapFile, flutterTestDataAttenFeaMapNum)
            # predhtAttenFeaMapFile.close()
            # '''颤振试飞试验实测数据验证'''

            '''风洞试验实测数据验证'''
            flutterTestDataFileResultCompairPath = mkSaveModelResultdir(
                curTrainResultSaveHomePath + '/' + windTunnelTestDataFileName)
            realDataDirPath = dataFilepath + '/' + windTunnelTestDataFileName
            fileNameList = os.listdir(realDataDirPath)
            print('fileNameList', fileNameList)

            for fileName in fileNameList:
                realDataFileName = fileName
                realDataFilePath = realDataDirPath + '/' + realDataFileName

                file = open(realDataFilePath, 'r')
                data = numpy.loadtxt(file)

                # print('----------------------data.shape',data.shape)
                multiCHTurSigNumpy = data
                data = data[:T* Fs, : ]
                # print('----------------------data.shape', data.shape)
                signal = numpy.reshape(data, [1, 1, 3, -1])
                # signal = (signal - numpy.mean(signal)) / numpy.std(signal)
                multiCHTurSig = dataAbout.numpyTOFloatTensor(signal)
                multiCHTurSig = multiCHTurSig.to(device)
                genImpRes, flutterTestDataAttenFeaMap = NetG(multiCHTurSig)

                MultiCHTurResFilePath = flutterTestDataFileResultCompairPath + "/" + realDataFileName
                numpy.savetxt(MultiCHTurResFilePath, multiCHTurSigNumpy)

                predhtFileName = realDataFileName[:len(realDataFileName) - 4] + '-predht.txt'
                predhtFilePath = flutterTestDataFileResultCompairPath + "/" + predhtFileName
                numpy.savetxt(predhtFilePath, genImpRes.cpu().detach().numpy().flatten())

                # attention feature map 保存
                predhtAttenFeaMapFileName = realDataFileName[:len(realDataFileName) - 4] + '-AttenFeaMap.txt'
                predhtAttenFeaMapFilePath = flutterTestDataFileResultCompairPath + "/" + predhtAttenFeaMapFileName

                flutterTestDataAttenFeaMapNumDim = flutterTestDataAttenFeaMap.size()

                flutterTestDataAttenFeaMapNum = \
                    numpy.zeros(shape=[flutterTestDataAttenFeaMapNumDim[2], flutterTestDataAttenFeaMapNumDim[3]])
                predhtAttenFeaMapFile = open(predhtAttenFeaMapFilePath, 'wb')
                for i in range(flutterTestDataAttenFeaMap.shape[1]):
                    flutterTestDataAttenFeaMappir = flutterTestDataAttenFeaMap[0, i, :, :]
                    flutterTestDataAttenFeaMappirNumpy = flutterTestDataAttenFeaMappir.cpu().detach().numpy()
                    flutterTestDataAttenFeaMapNum = \
                        numpy.concatenate((flutterTestDataAttenFeaMapNum, flutterTestDataAttenFeaMappirNumpy), axis=0)
                flutterTestDataAttenFeaMapNum = flutterTestDataAttenFeaMapNum[3:, :]
                numpy.savetxt(predhtAttenFeaMapFile, flutterTestDataAttenFeaMapNum)
            predhtAttenFeaMapFile.close()
            '''风洞试验实测数据验证结束'''

def mkSaveModelResultdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path


def main():
    response, timeSys, fileNameList = dataAbout.load_data(path, T * Fs);
    modelG = netG;
    modelD = netD
    # model = Transformer
    train_and_test(modelG, modelD, response, timeSys, fileNameList)


if __name__ == '__main__':
    dataFilepath = '/media/server/口..口/Duanshiqiang/multiCHAtten_GAN/data'

    # dataFilepath = '/media/server/口..口/Duanshiqiang/DLGenerateFRFHwHt/data';
    # dataFileName = 'tur1&2&3orderRandmodel_T-20s_Fs-512'
    # dataFileName = 'tur[2-4]Multimodels_Randmodel_T-20s_Fs-512'
    # dataFileName = 'tur[2-4]MulM_F[1-60]_D[0.005-0.1]_HNRand_20s_512'
    # dataFileName = 'tur[2-4]MulM_F[1-60]_D[0.005-0.1]_HNRand_20s_512'
    # dataFileName = 'tur[3-3]MulM_F[1-60]_D[0.005-0.1]_HNRand_20s_512'
    dataFileName = 'tur[2-4]Multimod_F[0.1-60]_D[0.005-0.2]_AmRand_20s_512'
    # dataFileName = 'tur[3-3]Multimod_F[0.1-60]_D[0.005-0.2]_AmRand_20s_512'
    # dataFileName = 'tur[2-4]MulM_F[1-60]_D[0.005-0.1]_HNRand_20s_512 - test'

    # dataFileName = 'tur[2-4]MulM_F[1-60]_D[0.005-0.1]_HNRand_20s_512 - test'

    flutterTestDataFileName = 'MultiCHTurRes_filtered'
    windTunnelTestDataFileName = 'MultiCHTurTes_windtunelDataResampled'

    path = dataFilepath + '/' + dataFileName
    # /media/server/口..口/Duanshiqiang/DLGenerateFRFHwHt/data/tur3order_T-20s_Fs-512
    ResultSaveHomePath = '/media/server/口..口/Duanshiqiang/multiCHAtten_GAN/result';

    MAXDATASIZE = 4500;
    TRAIN_TEST_RATE = 0.1;

    lossRate = 1

    BATCH_SIZE = int((MAXDATASIZE * (1 - TRAIN_TEST_RATE)) / 10);
    # BATCH_SIZE = 1
    T = 20;
    Fs = 512;

    NetGLayerNo = 10
    NetDLayerNo = 3
    NumberofFeatureChannel = 2;
    timeLength = T * Fs;

    embedding_dimension = 1
    encode_length = timeLength
    decode_length = timeLength

    ResCHNo = 1

    EPOCH = 20012;
    # ##kerne size =21###
    LR_G = 0.0001;
    LR_D = 0.0001

    main()





