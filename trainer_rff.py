# from src.models import *
from src.trainer import RFFTrainer, RFFConfs
from src.dyModels import LSTM_Attention_Model,FCModel,Transformer_Model,LSTM_MultiHeadAttention_Model,LSTM_FusionMultiheadAttention_Model,BiLSTM_Attention_Model
class Confs(RFFConfs):
    def __init__(self, train_snr=None, device=0, z_dim=64):
        super().__init__(train_snr, device, z_dim)
        
    def get_flag(self):
        self.eval_model = BiLSTM_Attention_Model
        self.data_idx = 0
        self.flag = 'BiLSTM-4_Attention_Model-60-RFF-ZDim-{}'.format(self.z_dim)
        self.epochs = 1000
        self.class_num =10
class Trainer(RFFTrainer, Confs):
    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='rff', train_snr=None, device=0, z_dim=512):
        Confs.__init__(self, train_snr, device, z_dim)
        RFFTrainer.__init__(self, config, ChannelNum, DeviceNum, regenerate_flag, label)
    def eval(self, epoch, eval_dataset='val', ext_name=''):
        return super().eval(epoch, eval_dataset, ext_name)

if __name__ == '__main__':
    import marveltoolbox as mt 
    config = {
        'power': [0.8, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [25, 35],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 30e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -25],  # 路径增益范围，单位dB
        'PathNum': [2, 3],  # 多径数量,
        'K':[15,20],
        'fd':[4,5]
    }
    # UserNum: 用户数量，默认60，按6:2:2划分
    # regenerate_flag: 生成数据集标志
    trainer = Trainer(device=0, z_dim=256, config=config, ChannelNum=60, DeviceNum=10, regenerate_flag=False, label='rff')
    params = mt.utils.params_count(trainer.models['C'])
    print(params)
    trainer.run(load_best=True, retrain=True, is_del_loger=False)
    trainer.eval(0, 'train')
    trainer.eval(0, 'val')