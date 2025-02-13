
from torchvision.datasets import MNIST
import torch.utils.data as data
from PIL import Image
from typing import Dict, Tuple
import collections
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from random import shuffle
import scipy
import marveltoolbox as mt
class RBdataset(torch.utils.data.Dataset):
    def __init__(self, data_path, id_list,SNR=None, rand_max_SNR=None, is_return_more=False,apply_channel_compensation=True):
        # mat_data = scipy.io.loadmat(data_path)

        # data = mat_data['data']
        # x = data[:, :-1]  # 取前12列作为特征
        # y =np.abs(data[:, -1])    # 取最后一列作为标签
        x, y  = propress(data_path,id_list,apply_channel_compensation)
        x_abs = np.abs(x)
        x_real = torch.tensor(x.real, dtype=torch.float32)  # 实部
        x_imag = torch.tensor(x.imag, dtype=torch.float32)  # 虚部
        x_abs = torch.tensor(x_abs.real, dtype=torch.float32)
        x_tensor = torch.stack((x_real, x_imag,x_abs), dim=-1)
        print(x_tensor.shape)
        y_tensor = torch.tensor(y-1, dtype=torch.long)

        self.data={}
        self.data['x_real']=x_real
        self.data['x_imag']=x_imag
        self.data['x_abs']=x_abs
        self.data['x']=x_tensor
        self.data['y']=y_tensor
        self.snr = SNR
        self.max_snr = rand_max_SNR
        self.is_return_more = is_return_more

    def __getitem__(self, index):
        x = self.data['x'][index]
        x = x.view(1,12,3)
        y = self.data['y'][index]
        return x,y

    def __len__(self):
        return len(self.data['y'])
    
def propress(data_path,id_list,appley_channel_compensation=True):
    mat_data = scipy.io.loadmat(data_path)
    mat_data['data_raw'].shape
    data = mat_data['data_raw'][:,:-2]
    user_id = np.abs(mat_data['data_raw'][:,-2]).astype(int)
    device_id = np.abs(mat_data['data_raw'][:,-1]).astype(int)
    data = data.reshape(data.shape[0],12,6)

    demod_bits = np.zeros((data.shape[0], data.shape[1], data.shape[2],2))
    demod_bits[:, :, :, 0] = 2 * (np.real(data) >= 0) - 1
    demod_bits[:, :, :,1] = 2 * (np.imag(data) >= 0) - 1
    symbol_value = (demod_bits[:, :, :, 0] + 1j * demod_bits[:, :, :, 1]) * (1 / np.sqrt(2))
    print(symbol_value.shape,data.shape)
    X = symbol_value
    Y = data
    H = np.zeros((6000, 6, 6), dtype=np.complex128)
    if appley_channel_compensation:
        # LS 信道估计计算
        for i in range(X.shape[0]):
            X_H = np.conjugate(X[i].T)  # 共轭转置 (6, 12)
            H[i] = np.linalg.inv(X_H @ X[i]) @ (X_H @ Y[i])  # 公式计算 (6, 6)
        corrected_X = np.zeros_like(Y) 
        for i in range(Y.shape[0]):
            corrected_X[i] = Y[i] @ np.linalg.inv(H[i])  # 去除信道影响
        Y= corrected_X
    CSI = Y /X
    CSI = np.mean(CSI, axis=-1)
    # plt.figure(figsize=(12, 8))
    # device_1_user_list = user_id[device_id==2]
    # user_id_list = np.unique(device_1_user_list)
    # for i in range(len(user_id_list)):
    #     device_rff = CSI[user_id == user_id_list[i],:]
    #     device_rff_mean = np.mean(device_rff, axis=0)
    #     plt.plot(np.abs(device_rff_mean), label=f"channel {i+1}")  # 每一行是一个折线

    # plt.xlabel("Index", fontsize=14)
    # plt.ylabel("Mean Value", fontsize=14)
    # plt.legend(loc='upper right', fontsize=8, ncol=2)  # 图例设置为两列，避免过多行
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('channel_plot.png', dpi=300)  # 设置dpi为300以获得高清图像
    print(np.isin(user_id, id_list))
    return CSI[np.isin(user_id, id_list)],device_id[np.isin(user_id, id_list)]
def load_RB_CSI(data_root,file_name,train_id_list,test_id_list,batch_size,pair=False,apply_channel_compensation=True):
    print('loading RB dataset...')
    data_path = os.path.join(data_root, 'RB_identify')
    file_path = os.path.join(data_path, file_name)
    train_dataset = RBdataset(file_path,train_id_list,apply_channel_compensation = apply_channel_compensation)
    test_dataset = RBdataset(file_path,test_id_list,apply_channel_compensation = apply_channel_compensation)
    print(f'train dataset size:{len(train_dataset)}')
    print(f'test dataset size:{len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
if __name__=='__main__':
    train_loader,test_loader = load_RB_CSI('./DATASET','RB_rff_data_60.mat',range(1,31),range(31,61),32)