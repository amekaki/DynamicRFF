import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from marveltoolbox.utils import TorchComplex as tc
# from .preprocessing import main
import scipy.io
import os

class RFdataset(torch.utils.data.Dataset):
    def __init__(self, data_file, SNR=None, rand_max_SNR=None, is_return_more=False):
        data_dir = '/workspace/DATASET/5G/RB_CSI/simulation'
        data_path = f"{data_dir}/{data_file}"
        mat_data = scipy.io.loadmat(data_path)

        data = mat_data['data']
        x = data[:, :-1]  # 取前12列作为特征
        y = data[:, -1]   # 取最后一列作为标签
        x = tc.array2tensor(x).float()
        y = tc.array2tensor(y)[:,0].view(-1).long()

        self.data={}
        self.data['x']=x
        self.data['y']=y
        self.snr = SNR
        self.max_snr = rand_max_SNR
        self.is_return_more = is_return_more
        
    def __getitem__(self, index):
        x        = self.data['x'][index]
        y        = self.data['y'][index]
        if not self.snr is None:
            x += tc.awgn(x, self.snr, SNR_x=30)
        
        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x += tc.awgn(x, rand_snr, SNR_x=30)
        return x,y

    def __len__(self):
        return len(self.data['y'])

if __name__ == "__main__":
    test = RFdataset(data_file='randomChannel_30device.mat')
    print(len(test))
    print(test[0][0].shape)
    print(test[0][1])