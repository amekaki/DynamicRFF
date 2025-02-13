from trainer_rff_classify import Trainer as rffTrainer

if __name__ == '__main__':
    import marveltoolbox as mt 
    config = {
        'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 5e-7],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 8]  # 多径数量
    }

    trainer = rffTrainer(device=1, z_dim=64, config=config, ChannelNum=6, DeviceNum=10, regenerate_flag=False, label='rff')
    params = mt.utils.params_count(trainer.models['C'])
    print(params)
    trainer.run(load_best=True, retrain=False, is_del_loger=False)
    trainer.eval(0)