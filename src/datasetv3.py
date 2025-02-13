import os
import torch
from torch.utils.data import DataLoader
import scipy.io
import torch.utils.data as data
import numpy as np
from .Gengrate_Dataset import *
import pandas as pd
from marveltoolbox.utils import TorchComplex as tc

class RFdataset(data.Dataset):
    def random(self,userId):
        userId=0
        # 遍历 waveform，index (device)
            # 遍历信道数量
                ## addChannel() 
                # userId+1, deviceId
        # 解调->数据集
        self.data['x']=0
        self.data['y_user']=0
        self.data['y_user']=0
        pass
    def __init__(self, file_name, config, UserNum=60, regenerate_flag=False,label = 'user'):
        '''
        参数: channel_num, config, label(user_classify/rff_identify) dataset ='test'
        '''
        # 生成或读取已经添加指纹的接入数据
        # self.list=[]
        # for deviceId in range(10):
        #   waveform，index = generateWaveform(deviceId)
        #   self.list.add(waveform，index,deviceId)

        file_name = f'{dataset}_x.csv'


        # 添加随机信道，并解调获取数据集
        # train = random_channel(waveform，index,channel_num)
        
        # 初始化：加指纹  waveform，index = generateWaveform()
        # getitem：加噪，加信道
        # 训练时帧内不叠加

        self.config = config
        self.UserNum = UserNum
        self.data={}
        data_dir = '/workspace/DATASET/5G/RB_CSI/simulation'
        data_path = f"{data_dir}/{file_name}"
        self.label = label
        print(regenerate_flag)
        # 如果不需要重新生成数据，且数据文件存在，则从文件中读取
        if not regenerate_flag and os.path.exists(data_path):
            print(f"Loading dataset from {data_path}...")
            df = pd.read_csv(data_path, header=None).astype('complex')
            # df_complex = df.map(complex)
            data = df.to_numpy()
        else:
            print("Generating new dataset...")
            generate_dataset(data_dir, config, UserNum)
            df = pd.read_csv(data_path, header=None).astype('complex')
            data = df.to_numpy()
        print(data.shape)
        x = data[:, :13]  # 取前12列作为特征
        y_user = data[:,-2]
        y_rff = data[:,-1]
        x = tc.array2tensor(x).float()
        self.data['x']=x
        self.data['y_rff'] = tc.array2tensor(y_rff)[:,0].view(-1).long()
        self.data['y_user'] = tc.array2tensor(y_user)[:,0].view(-1).long()

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, index):
        x = self.data['x'][index]
        if self.label=='user':            
            return x, self.data['y_user'][index]
        elif self.label == 'rff':
            return x, self.data['y_rff'][index]

def main():
    UserNum = 60       # 设备数量，默认60
    file_name = "TrainDataset.csv"
    regenerate_flag = False
    config = {
        'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [25, 35],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 30e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 5]  # 多径数量
    }

    dataset = RFdataset(file_name=file_name, config=config, UserNum=UserNum, regenerate_flag=regenerate_flag)

    # 使用 DataLoader 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 查看数据集大小
    print(f"数据集的大小: {len(dataset)}")

    # 获取一个批次的样本
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"x_batch shape: {x_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        print(f"Sample x: {x_batch[0]}")
        print(f"Sample y: {y_batch[0]}")
        break  # 只查看第一个批次

if __name__ == '__main__':
    main()