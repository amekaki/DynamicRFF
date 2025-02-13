
from torchvision.datasets import MNIST
import torch.utils.data as data
from PIL import Image
from typing import Dict, Tuple
import collections
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from marveltoolbox.utils import TorchComplex as tc
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from random import shuffle
import scipy
import marveltoolbox as mt

import random
import numpy as np

def create_pairs(data, labels, sample_num=4):
    """
    创建正负样本对，并限定每个类采样指定数量的正样本和负样本对。
    
    参数：
    - data: 输入的数据 (numpy array)
    - labels: 样本对应的标签 (numpy array)
    - sample_num: 每个类采样的正样本和负样本对的数量
    
    返回：
    - pairs: 样本对 (numpy array)
    - pair_labels: 样本对对应的标签 (numpy array)
    """
    pairs = []
    pair_labels = []
    unique_classes = np.unique(labels)

    for class_id in unique_classes:
        # 获取当前类的索引
        class_indices = np.where(labels == class_id)[0]
        
        # 获取与当前类不同的类的索引
        non_class_indices = np.where(labels != class_id)[0]
        for i in range(len(class_indices)):
            positive_samples = random.sample(list(class_indices), min(sample_num, len(class_indices)))
            negative_samples = random.sample(list(non_class_indices), min(sample_num, len(non_class_indices)))
            for j in range(len(positive_samples)):
                pairs.append((data[class_indices[i]], data[positive_samples[j]]))
                pairs.append((data[class_indices[i]], data[negative_samples[j]]))
                pair_labels.extend([1, 0])

        # # 随机选择 sample_num 个正样本对
        # positive_pairs = random.sample(list(class_indices), min(sample_num, len(class_indices)))
        # for i in range(len(positive_pairs)):
        #     for j in range(i + 1, len(positive_pairs)):
        #         pairs.append((data[positive_pairs[i]], data[positive_pairs[j]]))
        #         pair_labels.append(1)  # 正样本对标签为 1
        
        # # 随机选择 sample_num 个负样本对
        # negative_samples = random.sample(list(non_class_indices), min(sample_num, len(non_class_indices)))
        # for neg_sample in negative_samples:
        #     for pos_sample in positive_pairs:
        #         pairs.append((data[pos_sample], data[neg_sample]))
        #         pair_labels.append(0)  # 负样本对标签为 0

    return np.array(pairs), np.array(pair_labels)



class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        pairs, pair_labels = create_pairs(data, labels)
        self.pairs = tc.array2tensor(pairs).float()
        self.labels =  torch.tensor(pair_labels, dtype=torch.long)

    def __getitem__(self, index):
        x1, x2 = self.pairs[index]
        label = self.labels[index]
        x1 = x1.view(1, 12, 3)  # 重新调整形状
        x2 = x2.view(1, 12, 3)
        return x1, x2, label

    def __len__(self):
        return len(self.labels)
    
class RBdataset(torch.utils.data.Dataset):
    def __init__(self, data_path, id_list,SNR=None, rand_max_SNR=None, is_return_more=False,apply_channel_compensation=True,load_pair = False):
        x, y  = propress(data_path,id_list,apply_channel_compensation)
        # x_abs = np.abs(x)
        # x_real = torch.tensor(x.real, dtype=torch.float32)  # 实部
        # x_imag = torch.tensor(x.imag, dtype=torch.float32)  # 虚部
        # x_abs = torch.tensor(x_abs.real, dtype=torch.float32)
        # x_tensor = torch.stack((x_real, x_imag,x_abs), dim=-1)
        # y_tensor = torch.tensor(y-1, dtype=torch.long)
        if load_pair:
            pairDataset = PairDataset(x,y)
            x_tensor = pairDataset.pairs
            y_tensor = pairDataset.labels
        else:
            x_tensor = tc.array2tensor(x).float()
            y_tensor = tc.array2tensor(y)[:,0].view(-1).long()
        self.load_pair = load_pair
        self.data={}
        self.data['x']=x_tensor
        self.data['y']=y_tensor
        self.snr = SNR
        self.max_snr = rand_max_SNR
        self.is_return_more = is_return_more

    def __getitem__(self, index):
        if self.load_pair:
            x1, x2 = self.data['x'][index]
            label = self.data['y'][index]
            return x1, x2, label
        x = self.data['x'][index]
        y = self.data['y'][index]
        return x,y

    def __len__(self):
        return len(self.data['y'])
    
def propress(data_path,id_list,appley_channel_compensation=True):
    mat_data = scipy.io.loadmat(data_path)
    mat_data['data_raw'].shape
    
    data = mat_data['data_raw'][:,:-2]
    user_id = np.abs(mat_data['data_raw'][:,-2]).astype(int)-1
    device_id = np.abs(mat_data['data_raw'][:,-1]).astype(int)-1
    data = data.reshape(data.shape[0],12,6)
    data_size = data.shape[0]
    demod_bits = np.zeros((data.shape[0], data.shape[1], data.shape[2],2))
    demod_bits[:, :, :, 0] = 2 * (np.real(data) >= 0) - 1
    demod_bits[:, :, :,1] = 2 * (np.imag(data) >= 0) - 1
    symbol_value = (demod_bits[:, :, :, 0] + 1j * demod_bits[:, :, :, 1]) * (1 / np.sqrt(2))
    print(symbol_value.shape,data.shape)
    X = symbol_value
    Y = data
    H = np.zeros((data_size, 6, 6), dtype=np.complex128)
    if appley_channel_compensation:
        for i in range(X.shape[0]):
            X_H = np.conjugate(X[i].T) 
            H[i] = np.linalg.inv(X_H @ X[i]) @ (X_H @ Y[i]) 
        corrected_X = np.zeros_like(Y) 
        for i in range(Y.shape[0]):
            corrected_X[i] = Y[i] @ np.linalg.inv(H[i]) 
        Y= corrected_X
    CSI = Y /X
    CSI = np.mean(CSI, axis=-1)
    print(np.isin(user_id, id_list))
    return CSI[np.isin(device_id, id_list)],device_id[np.isin(device_id, id_list)]

def load_RB_CSI(data_root,file_name,train_id_list,test_id_list,batch_size,load_pair=False,apply_channel_compensation=True):
    print('loading RB dataset...')
    data_path = os.path.join(data_root, 'RB_identify')
    file_path = os.path.join(data_path, file_name)
    train_dataset = RBdataset(file_path,train_id_list,apply_channel_compensation = apply_channel_compensation,load_pair=load_pair)
    test_dataset = RBdataset(file_path,test_id_list,apply_channel_compensation = apply_channel_compensation,load_pair=load_pair)
    print(f'train dataset size:{len(train_dataset)}')
    print(f'test dataset size:{len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset,test_dataset

if __name__=='__main__':
    train_loader,test_loader = load_RB_CSI('/workspace/DATASET/','RB_rff_data_60.mat',range(1,31),range(31,61),32,load_pair=False)