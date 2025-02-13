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
            torch.cuda.is_available() else "mps")


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
        self.datasets['val'] = RFdataset(config=config, ChannelNum=3, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='val')
        self.datasets['test'] = RFdataset(config=config, ChannelNum=3, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='test')
        self.datasets['close'] = RFdataset(config=config, ChannelNum=3, DeviceNum=DeviceNum, regenerate_flag=regenerate_flag, label=label, datasetname='close')

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
            x, y = data
            # print(np.unique(y))
            x, y = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x, y)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        # self.dataloaders['train'].random()
        return loss.item()
                
    def eval(self, epoch, eval_dataset = 'val',ext_name=''):
        self.logs = {}
        if(self.label == 'rff' and eval_dataset =='val'):
            eval_dataset = 'close'
        self.models['C'].eval()
        correct = 0.0
        test_loss = 0.0
        feature_list = []
        label_list = []
        print(eval_dataset,"eval_dataset")
        with torch.no_grad():
            for data in self.dataloaders[eval_dataset]:
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                features = self.models['C'].features(x)
                feature_list.append(features)
                label_list.append(y)

                if eval_dataset == 'close':
                    # features = self.models['C'].features(x)
                    scores = self.models['C'].output(features)
                    test_loss += F.cross_entropy(scores, y, reduction='sum').item()
                    pred_y = torch.argmax(scores, dim=1)
                    correct += torch.sum(pred_y == y).item()

        features = torch.cat(feature_list, dim=0)
        labels = torch.cat(label_list)
        intra_dist, inter_dist = inter_intra_dist(features.cpu().detach().numpy(), labels.cpu().numpy())
        distance_hist_plot(intra_dist, inter_dist, filename='./plots/{}_{}_dist_hist{}.png'.format(self.flag, eval_dataset,ext_name))
        eer, roc_auc, thresh = get_auc_eer(intra_dist, inter_dist, plot_roc=True, filename='./plots/{}_{}_roc{}.png'.format(self.flag, eval_dataset,ext_name))
        is_best = False

        if eval_dataset == 'close':
            acc = correct / len(self.datasets[eval_dataset])
            test_loss = test_loss/ len(self.datasets[eval_dataset])
                
            if acc >= self.records['acc']:
                is_best = True
                self.records['acc'] = acc
            self.logs['Test Loss'] = test_loss
            self.logs['acc'] = acc
            self.logs['auc'] = roc_auc
            self.logs['eer'] = eer
            self.logs['data'] = eval_dataset
            self.print_logs(epoch, 0)
        else:

            if roc_auc >= self.records['auc']:
                is_best = True
                self.records['auc'] = roc_auc
            self.logs['auc'] = roc_auc
            self.logs['eer'] = eer
            self.logs['data'] = eval_dataset
            self.print_logs(epoch, 0)
        
        return is_best
