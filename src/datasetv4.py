import os
import torch
from torch.utils.data import DataLoader
import scipy.io
import torch.utils.data as data
import numpy as np
from .Gengrate_Dataset import *
import pandas as pd
from marveltoolbox.utils import TorchComplex as tc
import random
DATASET_DIR = '/home/workplace/data/'
class RFdataset(data.Dataset):
    
    def random_channel(self, ChannelNum, DeviceNum, config):
        # 生成配置信息
        deviceConfigList = generate_device_config(ChannelNum*DeviceNum, config)
        print(deviceConfigList)
        label_size = 2
        data_size = 13 + label_size
        dataset = np.empty((0, data_size))
        UserId = 0
        dyRB_x_list = []
        dyRB_y_list = []
        dyRB_len_list = []
        for DeviceID in range(DeviceNum):
            for Channel in range(ChannelNum):
                # 加信道
                # print(len(self.list),DeviceID,self.list[DeviceID])
                rx_signal = multipath_rice_fading(self.list[DeviceID][0][0], deviceConfigList[UserId])
                # 加噪
                rx_signal = awgn(rx_signal, deviceConfigList[UserId])
                # 解资源格
                resourceGrid = showFreqDomain(rx_signal)
                # 生成数据集
                RB_CSI_FFT, RB_CSI = extractData(resourceGrid, self.list[DeviceID][1][0]-1, UserId, DeviceID)
                # print("shape",RB_CSI.shape)    
                # RB_CSI = torch.from_numpy(RB_CSI)
                dataset = np.vstack((dataset, RB_CSI_FFT))
                dyRB_x_list.append(np.pad(RB_CSI[:10], ((0, 30), (0, 0)), mode='constant', constant_values=0))
                dyRB_x_list.append(np.pad(RB_CSI[:20], ((0, 20), (0, 0)), mode='constant', constant_values=0))
                dyRB_x_list.append(np.pad(RB_CSI[:30], ((0, 10), (0, 0)), mode='constant', constant_values=0))
                dyRB_x_list.append(RB_CSI[:40])
                dyRB_y_list.append(DeviceID)
                dyRB_y_list.append(DeviceID)
                dyRB_y_list.append(DeviceID)
                dyRB_y_list.append(DeviceID)
                dyRB_len_list.append(10)
                dyRB_len_list.append(20)
                dyRB_len_list.append(30)
                dyRB_len_list.append(40)
                UserId = UserId + 1
        x = dataset[:, :13]
        y_user = dataset[:,-2]
        y_rff = dataset[:,-1]
        # print(np.unique(y_user))
        # print(np.unique(y_rff))
        self.data['x'] = torch.from_numpy(np.asarray(dyRB_x_list)).float()
        self.data['y'] = torch.from_numpy(np.asarray(dyRB_y_list))
        self.data['len'] = torch.from_numpy(np.asarray(dyRB_len_list))
        self.data['y_rff'] = tc.array2tensor(y_rff)[:,0].view(-1).long()
        self.data['y_user'] = tc.array2tensor(y_user)[:,0].view(-1).long()
        return dataset

    def generateWaveform(self, ChannelNum, DeviceNum):
        # 加载理想信号
        SignalConfig = scipy.io.loadmat(DATASET_DIR+'/DeviceIdealSignal_2.mat', squeeze_me=True)['DeviceIdealSignal'][0:DeviceNum*ChannelNum]
        validgirdindex = SignalConfig['index']
        IdealSignal = SignalConfig['IdealTimeDomainSignal']
        IdealSignal = np.vstack(IdealSignal)   # 展开堆叠
        validgirdindex = np.vstack(validgirdindex)
        # 加指纹
        self.list = []
        for deviceId in range(DeviceNum):
            waveform_index = np.random.choice(range(len(IdealSignal)), size=ChannelNum, replace=True)
            waveform = addRFF(IdealSignal[waveform_index], deviceId)
            index = validgirdindex[waveform_index]
            self.list.append([waveform, index])
            # print(waveform.shape)
            # print(deviceId,self.list[deviceId],IdealSignal[deviceId*ChannelNum:(deviceId+1)*ChannelNum])

    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label = 'user', datasetname ='train'):
        
        self.label = label
        self.data = {}
        self.ChannelNum = ChannelNum
        self.config =config
        self.DeviceNum = DeviceNum
        self.datasetname = datasetname
        data_dir = DATASET_DIR+'/dynamic/'
        data_path = f"{data_dir}/{datasetname}dataset.csv"
        dy_x_path = f"{data_dir}/{datasetname}_x.pt"
        dy_y_path = f"{data_dir}/{datasetname}_y.pt"
        dy_len_path = f"{data_dir}/{datasetname}_len.pt"
        if not regenerate_flag and os.path.exists(dy_x_path):           # 读取
            print(f"Loading dataset from {dy_x_path}...")
            df = pd.read_csv(data_path, header=None).astype('complex')
            data = df.to_numpy()
            x = data[:, :13]
            y_user = data[:,-2]
            y_rff = data[:,-1]
            self.data['x'] = tc.array2tensor(x).float()
            self.data['y_rff'] = tc.array2tensor(y_rff)[:,0].view(-1).long()
            self.data['y_user'] = tc.array2tensor(y_user)[:,0].view(-1).long()
            self.data['x'] = torch.load(dy_x_path)
            self.data['y'] = torch.load(dy_y_path)
            self.data['len'] = torch.load(dy_len_path)
        else:# 生成并保存
            print("Generating new dataset...")
            # 生成或读取已经添加指纹的接入数据
            self.generateWaveform(ChannelNum, DeviceNum)
            # 加噪，加信道，并解调生成数据集
            dataset = self.random_channel(ChannelNum, DeviceNum, config)
            dataset_pd = pd.DataFrame(dataset)
            dataset_pd.to_csv(data_path, index=False, header=False)
            torch.save(self.data['x'], dy_x_path)
            torch.save(self.data['y'], dy_y_path)
            torch.save(self.data['len'], dy_len_path)

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, index):
        x = self.data['x'][index]
        y = self.data['y'][index]
        x_len = self.data['len'][index]
        return x,y,x_len
        if self.label=='user':            
            return x, self.data['y_user'][index]
        elif self.label == 'rff':
            return x, self.data['y_rff'][index]

def main():
    config = {
        'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 30e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 8]  # 多径数量
    }
    dataset = RFdataset(config=config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='rff', datasetname='train')
    # print(dataset[101])
    # DeviceNum = 6
    # ChannelNum = 10
    # SignalConfig = scipy.io.loadmat('/workspace/DATASET/5G/RB_CSI/simulation/DeviceIdealSignal.mat', squeeze_me=True)['DeviceIdealSignal'][0:DeviceNum*ChannelNum]
    # validgirdindex = SignalConfig['index']
    # IdealSignal = SignalConfig['IdealTimeDomainSignal']
    # IdealSignal = np.vstack(IdealSignal)
    # validgirdindex = np.vstack(validgirdindex)
    # print(IdealSignal.shape)         # 展开堆叠
    # print(validgirdindex.shape)
    # config = {
    #     'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
    #     'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
    #     'PathDelay': [0, 30e-6],  # 路径延迟范围
    #     'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
    #     'PathNum': [2, 8]  # 多径数量
    # }

    # list = []
    # for deviceId in range(DeviceNum):
    #     waveform = addRFF(IdealSignal[deviceId*ChannelNum:(deviceId+1)*ChannelNum,:], deviceId)
    #     index = validgirdindex[deviceId*ChannelNum:(deviceId+1)*ChannelNum,:]
    #     list.append([waveform, index])
    # print(len(list))
    # print(list[0][0].shape)
    # print(list[0][1].shape)
        
    # deviceConfigList = generate_device_config(ChannelNum*DeviceNum, config)
    # print(len(deviceConfigList))
    # label_size = 2
    # data_size = 13 + label_size
    # dataset = np.empty((0, data_size))
    # UserId = 0
    # for DeviceID in range(DeviceNum):
    #     for Channel in range(ChannelNum):
    #         # 加信道
    #         rx_signal = addChannel(list[DeviceID][0][Channel], deviceConfigList[UserId])
    #         # 加噪
    #         rx_signal = awgn(rx_signal, deviceConfigList[UserId])
    #         # 解资源格
    #         resourceGrid = showFreqDomain(rx_signal)
    #         # 生成数据集
    #         RB_CSI_FFT, RB_CSI = extractData(resourceGrid, list[DeviceID][1][Channel]-1, UserId+1, DeviceID+1)
    #         dataset = np.vstack((dataset, RB_CSI_FFT))
    #         UserId = UserId + 1
    # print(dataset.shape)
    # # dataset_pd = pd.DataFrame(dataset)
    # # dataset_pd.to_csv('/workspace/DATASET/5G/RB_CSI/simulation/TrainDataset.csv', index=False, header=False)
    # x = dataset[:, :13]
    # y_user = dataset[:,-2]
    # y_rff = dataset[:,-1]

    # x = abs(x[0:1000,:])
    # y = abs(y_user[0:1000])
    # tsne = TSNE(n_components=2, random_state=42)
    # data_tsne = tsne.fit_transform(x)
    # plot_tsne_2D(data_tsne, y)

if __name__ == '__main__':
    main()