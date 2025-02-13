# from src.models import *
from src.trainer import RFFTrainer, RFFConfs
from src.dyModels import LSTM_Attention_Model,FCModel,Transformer_Model,LSTM_MultiHeadAttention_Model,LSTM_FusionMultiheadAttention_Model,BiLSTM_Attention_Model
config = {
    'channelName': 'los2',
    'modelName': 'BiLSTM-Attention',
    'DeviceNum':10,
    'ChannelNum':60,
    'z_dim':256
}
class Confs(RFFConfs):
    def __init__(self, config,train_snr=None, device=0, z_dim=64):
        super().__init__(train_snr, device, z_dim)
        
    def get_flag(self):
        self.eval_model = BiLSTM_Attention_Model
        self.data_idx = 0
        self.flag = '{}-channel-{}-num-{}-ZDim-{}'.format(config['modelName'],config['channelName'],config['ChannelNum'],config['z_dim'])
        self.epochs = 400
        self.class_num =10
class Trainer(RFFTrainer, Confs):
    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='rff', train_snr=None, device=0, z_dim=512):
        Confs.__init__(self,config, train_snr, device, z_dim)
        RFFTrainer.__init__(self, config, ChannelNum, DeviceNum, regenerate_flag, label)
    def eval(self, epoch, eval_dataset='val', ext_name=''):
        return super().eval(epoch, eval_dataset, ext_name)

if __name__ == '__main__':
    import marveltoolbox as mt 
    # UserNum: 用户数量，默认60，按6:2:2划分
    # regenerate_flag: 生成数据集标志
    trainer = Trainer(device=0, z_dim=config['z_dim'], config=config, ChannelNum=config['ChannelNum'], DeviceNum=config['DeviceNum'], regenerate_flag=False, label='rff')
    params = mt.utils.params_count(trainer.models['C'])
    print(params)
    trainer.run(load_best=True, retrain=True, is_del_loger=False)
    trainer.eval(0, 'train')
    trainer.eval(0, 'val')