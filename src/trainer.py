import sys
import marveltoolbox as mt 
from .models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .datasetv4 import *
from .evaluation import *
from .RBDataset import load_RB_CSI
class RFFConfs(mt.BaseConfs):
    def __init__(self, train_snr, device=0, z_dim=32):
        self.train_snr = train_snr
        self.device = device
        self.device_ids = [device]
        self.z_dim = z_dim
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'zigbee'
        self.batch_size = 64
        self.class_num = 10
        self.epochs = 10
        self.train_devices = range(45)
        self.train_ids = [1,2,3,4]

    def get_flag(self):
        self.eval_model = CSI_CLF
        self.data_idx = 0
        self.flag = 'Baseline-UserOpenset-ZDim{}'.format(self.z_dim)
    
    def get_device(self):
        self.device = torch.device(
            "cuda:{}".format(self.device) if \
            torch.cuda.is_available() else "cpu")
        print(self.device)


class RFFTrainer(mt.BaseTrainer):
    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='user'):
        mt.BaseTrainer.__init__(self, self)
        if(label=='user'):
            self.class_num = ChannelNum*DeviceNum
        else:
            self.class_num = DeviceNum
        self.label = label
        
        self.models['C'] = self.eval_model(z_dim=self.z_dim, classes=self.class_num).to(self.device)

        self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))

        self.datasets['train'] = RFdataset(config=config, ChannelNum=ChannelNum, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='train')
        self.datasets['val'] = RFdataset(config=config, ChannelNum=ChannelNum, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='val')
        # self.datasets['test'] = RFdataset(config=config, ChannelNum=2, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='test')
        # self.datasets['close'] = RFdataset(config=config, ChannelNum=3, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='close')
        
        # self.datasets['train'],self.datasets['val'] = load_RB_CSI('/workspace/DATASET/','RB_rff_data_60.mat',range(0,3),range(3,6),32,load_pair=False)
        # self.datasets['train'] = RFdataset(data_file='randomChannel_30device.mat', SNR=self.train_snr)
        # self.datasets['val'] = RFdataset(data_file='randomChannel_10device.mat')
        # self.datasets['close'] = RFdataset(data_file='randomChannel_10device.mat')
        # self.datasets['close'] = RFdataset(device_ids=range(45), test_ids=[5], rand_max_SNR=None)

        self.preprocessing()

        self.records['acc'] = 0.0
        self.records['auc'] = 0.0

    def train(self, epoch):
        self.logs = {}
        self.models['C'].train()
        for i, data in enumerate(self.dataloaders['train']):
            x, y,x_len = data
            # print(np.unique(y))
            x, y = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x, x_len)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            # print(scores)
            # print(y,x_len)
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        # self.dataloaders['train'].random()
        return loss.item()
                
    def eval(self, epoch, eval_dataset = 'close',ext_name=''):
        self.logs = {}
        # if(self.label == 'rff' and eval_dataset =='val'):
        #     eval_dataset = 'close'
        self.models['C'].eval()
        correct = 0.0
        total_num=0
        test_loss = 0.0
        feature_list = []
        label_list = []
        print(eval_dataset,"eval_dataset")
        print(self.dataloaders[eval_dataset].__len__())
        with torch.no_grad():
            for data in self.dataloaders[eval_dataset]:
                # print("load")
                x, y,x_len = data
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                scores = self.models['C'](x,x_len)
                # print(scores)
                # print(y,x_len)
                test_loss += F.cross_entropy(scores, y, reduction='sum').item()
                pred_y = torch.argmax(scores, dim=1)
                correct += torch.sum(pred_y == y).item()
                total_num += len(pred_y)  
        is_best = False
        acc = correct / total_num
        test_loss = test_loss/ len(self.datasets[eval_dataset])
            
        if acc >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc
        self.logs['Test Loss'] = test_loss
        self.logs['acc'] = acc
        self.logs['data'] = eval_dataset
        self.print_logs(epoch, 0)
        
        
        return is_best
