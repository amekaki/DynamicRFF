from src.models import *
from src.trainer import RFFTrainer, RFFConfs

class Confs(RFFConfs):
    def __init__(self, train_snr=None, device=0, z_dim=64):
        super().__init__(train_snr, device, z_dim)
        
    def get_flag(self):
        self.eval_model = CSI_CLF
        self.data_idx = 0
        self.flag = 'Baseline-UserOpenset-ZDim-power-K-{}'.format(self.z_dim)

class Trainer(RFFTrainer, Confs):
    def __init__(self, config, ChannelNum=10, DeviceNum=6, regenerate_flag=False, label='user', train_snr=None, device=0, z_dim=512):
        Confs.__init__(self, train_snr, device, z_dim)
        RFFTrainer.__init__(self, config, ChannelNum, DeviceNum, regenerate_flag, label)

if __name__ == '__main__':
    import marveltoolbox as mt 
    config = {
        'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 5e-7],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 8]  # 多径数量
    }
    ChannelNum=10
    DeviceNum=6
    trainer = Trainer(device=1, z_dim=64, config=config, ChannelNum=ChannelNum, DeviceNum=DeviceNum, regenerate_flag=False, label='user')
    trainer.class_num = ChannelNum*DeviceNum
    print(trainer.class_num)
    trainer.epochs = 200
    params = mt.utils.params_count(trainer.models['C'])
    print(params)
    trainer.run(load_best=True, retrain=True, is_del_loger=False)
    trainer.eval(0, 'val')