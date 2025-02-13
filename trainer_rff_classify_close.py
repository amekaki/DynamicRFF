from src.models import *
from src.trainer import RFFTrainer, RFFConfs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
class Confs(RFFConfs):
    def __init__(self, train_snr=None, device=0, z_dim=64):
        super().__init__(train_snr, device, z_dim)
        
    def get_flag(self):
        self.eval_model = FCModel
        self.data_idx = 0
        self.flag = 'Baseline-RFFClassify-ZDim-random-index-close{}'.format(self.z_dim)
        self.epochs =200

class Trainer(RFFTrainer, Confs):
    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='rff', train_snr=None, device=0, z_dim=512):
        Confs.__init__(self, train_snr, device, z_dim)
        RFFTrainer.__init__(self, config, ChannelNum, DeviceNum, regenerate_flag, label)
        self.records['acc'] = 0.0
        self.records['loss'] = 1

    def train(self, epoch):
        self.models['C'].train()
        for i, (x, y) in enumerate(self.dataloaders['train']):
            x, y = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x)
            # print(scores.shape)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            if i % 1000 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)
        # self.dataloaders['train'].random()
        return loss.item()
                
    def eval(self, epoch):
        self.models['C'].eval()
        correct = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in self.dataloaders['close']:
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                scores = self.models['C'](x)
                pred_y = torch.argmax(scores, dim=1)
                correct += torch.sum(pred_y == y).item()
                all_preds.append(pred_y.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        N = len(self.dataloaders['close'])
        acc = correct / N
        is_best = False
        if acc >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc
            self.plot_confusion_matrix(all_labels, all_preds, epoch)
            print('acc: {}'.format(acc))
        else:
            print('acc/best acc: {}/{}'.format(acc,self.records['acc']))
        return is_best

    def plot_confusion_matrix(self, all_labels, all_preds, epoch):
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
            plt.title(f'Confusion Matrix at Epoch {epoch}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig(f'confusion_matrix-{self.flag}.png')
if __name__ == '__main__':
    import marveltoolbox as mt 
    config = {
        'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 5e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 8]  # 多径数量
    }

    trainer = Trainer(device=1, z_dim=64, config=config, ChannelNum=10, DeviceNum=10, regenerate_flag=False, label='rff')
    params = mt.utils.params_count(trainer.models['C'])
    print(params)
    trainer.run(load_best=True, retrain=True, is_del_loger=False)
    trainer.eval(0)