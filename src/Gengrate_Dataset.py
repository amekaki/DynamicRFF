import numpy as np
import scipy.io
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# 加噪
def awgn(tx_waveform, config):
    snr_db = config['SNRdB']
    np.random.seed(42)
    # 计算信号功率
    signal_power = np.mean(np.abs(tx_waveform)**2)
    # 计算噪声功率
    noise_power = signal_power / (10**(snr_db / 10))
    # 生成噪声，利用标准差对标准正态分布进行缩放
    noise = np.sqrt(noise_power) * np.random.randn(*tx_waveform.shape)
    # 将噪声添加到信号中
    rx_waveform = tx_waveform + noise

    return rx_waveform

# 加指纹
def addRFF(tx_signal, id, fs=61440000):
    # 设备参数 (频率偏移, 增益不平衡, 相位不平衡)
    device_params = np.array([
        [10, 0.98, 1 * np.pi / 180],
        [20, 0.95, 2 * np.pi / 180],
        [30, 0.90, 3 * np.pi / 180],
        [40, 0.93, 4 * np.pi / 180],
        [50, 1, 5 * np.pi / 180],
        [60, 0.75, 10 * np.pi / 180],
        [70, 0.88, 6 * np.pi / 180],
        [80, 0.85, 7 * np.pi / 180],
        [90, 0.83, 8 * np.pi / 180],
        [100, 0.80, 9* np.pi / 180]
    ])

    # id从1开始
    # id = (id ) % 6
    # 获取当前设备的参数
    freq_offset = device_params[id, 0]  # 频率偏移
    gain_imbalance = device_params[id, 1]  # 增益不平衡
    phase_imbalance = device_params[id, 2]  # 相位不平衡

    # 信号长度
    signal_rows, signal_cols = tx_signal.shape
    # print(signal_length)
    # 时间向量 (秒)
    t = np.tile(np.arange(signal_cols) / fs, (signal_rows, 1))
    # 添加频率偏移
    tx_signal_of = tx_signal * np.exp(1j * 2 * np.pi * freq_offset * t)
    # 添加增益不平衡和相位不平衡
    real_part = np.real(tx_signal_of) * gain_imbalance
    imag_part = np.imag(tx_signal_of) * np.cos(phase_imbalance) + np.real(tx_signal_of) * np.sin(phase_imbalance)
    # 生成接收信号
    rx_signal = real_part + 1j * imag_part

    return rx_signal

# 加信道
def addChannel(txSignal, config, fs=61440000):
    pathDelays = config['pathDelays']
    pathGains = config['avgPathGains']
    # 初始化接收波形
    rxWaveform = np.zeros_like(txSignal)
    # 遍历每条多径
    for i in range(len(pathDelays)):
        # 计算延迟对应的采样数
        delaySamples = round(pathDelays[i] * fs)
        # 添加0以实现时延，并乘以路径增益（将dB转换为线性增益）
        attenuatedSignal = np.concatenate([np.zeros(delaySamples), txSignal]) * 10 ** (pathGains[i] / 20)
        # 多条路径叠加，得到最终的多径叠加信号
        rxWaveform += attenuatedSignal[:len(txSignal)]

    return rxWaveform

def generate_rice_fading(N, K, fd, fs):
    """
    生成单条路径的莱斯衰落信号
    :param N: 信号长度
    :param K: 莱斯因子（线性值）
    :param fd: 最大多普勒频移（Hz）
    :param fs: 采样率（Hz）
    :return: 单路径莱斯衰落信号
    """

    t = np.arange(N) / fs
    
    # LOS分量
    A = np.sqrt(K / (K + 1))  # LOS 分量的幅度
    los_component = A * np.exp(1j * 2 * np.pi * fd * t)  # 加入多普勒频移的相位变化

    # 生成 NLOS 分量（动态瑞利衰落）
    N_c = 8  # 离散正弦波数量
    theta = np.random.uniform(0, 2 * np.pi, N_c)  # 随机相位
    doppler_frequencies = fd * np.cos(2 * np.pi * np.arange(1, N_c + 1) / N_c)
    
    i_component = np.sum(np.cos(2 * np.pi * np.outer(doppler_frequencies, t) + theta[:, None]), axis=0)
    q_component = np.sum(np.sin(2 * np.pi * np.outer(doppler_frequencies, t) + theta[:, None]), axis=0)
    nlos_component = (i_component + 1j * q_component) / np.sqrt(2 * N_c)  # 瑞利分量归一化

    # 调整 NLOS 分量的功率
    nlos_component *= np.sqrt(1 / (K + 1))
    
    # 组合 LOS 和 NLOS 分量
    rice_fading_signal = los_component + nlos_component
    return rice_fading_signal

def multipath_rice_fading(signal, config, fs=61440000):
    """
    模拟多径莱斯衰落信道
    :param signal: 输入信号
    :param K: 莱斯因子（dB 或线性值）
    :param fd: 最大多普勒频移（Hz）
    :param fs: 信号采样率（Hz）
    :param path_delays: 多径延迟列表（秒）
    :param path_gains: 多径增益列表（线性比例）
    :return: 加入多径莱斯衰落的信号
    """
    # K, fd, path_delays, path_gains
    K = config['K']
    fd = config['fd']
    path_delays = config['pathDelays']
    path_gains = config['avgPathGains']
    # 将 K 因子从 dB 转为线性值（若已是线性值则忽略此步骤）
    if K > 0 and K < 30:  # 假定 K 因子范围为 dB 时
        K = 10 ** (K / 10)
    
    N = len(signal)  # 输入信号的长度
    path_samples = [int(delay * fs) for delay in path_delays]  # 延迟转化为采样点数
    multipath_signal = np.zeros(N, dtype=complex)  # 初始化多径信号

    for delay, gain in zip(path_samples, path_gains):
        # 生成单路径的莱斯衰落信号
        fading_path = generate_rice_fading(N, K, fd, fs)
        # 延迟信号并加入增益
        delayed_signal = np.zeros(N, dtype=complex)
        delayed_signal[delay:] = signal[:N - delay] * fading_path[:N - delay] * 10 ** (gain / 10)
        # 累加多径信号
        multipath_signal += delayed_signal

    return multipath_signal

# 生成设备配置列表
def generate_device_config(DeviceNum, path_config):
    # 初始化设备配置列表
    deviceConfigList = []
    # 为每台设备生成配置
    for id in range(1, DeviceNum + 1):
        # 设备配置字典
        device = {}
        # 随机生成信噪比 (SNR) 和功率
        device['id'] = id
        device['SNRdB'] = np.random.randint(path_config['SNR'][0], path_config['SNR'][1] + 1)  # SNR 随机生成
        device['power'] = path_config['power'][0] + (
                    path_config['power'][1] - path_config['power'][0]) * np.random.rand()  # 功率随机生成

        # 随机生成路径数目
        numPaths = np.random.randint(path_config['PathNum'][0], path_config['PathNum'][1] + 1)  # 随机生成路径数目

        # 随机生成路径延迟和路径增益
        pathDelays = np.sort(path_config['PathDelay'][0] +
                             (path_config['PathDelay'][1] - path_config['PathDelay'][0]) * np.random.rand(numPaths))
        avgPathGains = path_config['AvgPathGain'][0] + \
                       (path_config['AvgPathGain'][1] - path_config['AvgPathGain'][0]) * np.random.rand(numPaths)

        # 测试用
        # pathDelays = np.sort(path_config['PathDelay'][0] +
        #                      (path_config['PathDelay'][1] - path_config['PathDelay'][0]) * np.array([0.135,0.367,0.872]))
        # avgPathGains = path_config['AvgPathGain'][0] + \
        #                (path_config['AvgPathGain'][1] - path_config['AvgPathGain'][0]) * np.array([0.235,0.467,0.772])

        # 设置路径延迟和增益
        pathDelays[0] = 0  # 第一条路径延迟为 0
        avgPathGains[0] = 0  # 第一条路径增益为 0

        # 将路径延迟和路径增益存储到设备配置中
        device['pathDelays'] = pathDelays
        device['avgPathGains'] = avgPathGains
        fd_range = (path_config['fd'][0],path_config['fd'][1])
        # 随机生成最大多普勒频移 fd
        fd = np.random.uniform(*fd_range)  # 低速场景，fd 较小
        device['fd'] = fd
        # 随机生成莱斯因子 K
        K_range = (path_config['K'][0],path_config['K'][1])
        K = np.random.uniform(*K_range)  # K 因子通常较大（dB）
        device['K'] = K
        # 将设备配置添加到设备配置列表中
        deviceConfigList.append(device)

    return deviceConfigList

def generate_dataset(data_dir, config, DeviceNum = 60,file_name = 'TrainDataset.csv'):
    label_size = 2
    data_size = 13+label_size
    dataset = np.empty((0, data_size))
    dataset_train = np.empty((0, data_size))
    dataset_val = np.empty((0, data_size))
    dataset_test = np.empty((0, data_size))
    # --------------------------------加载理想信号---------------------------------------------------
    file_name = 'DeviceIdealSignal.mat'
    data_path = f"{data_dir}/{file_name}"
    SignalConfig = scipy.io.loadmat(data_path, squeeze_me=True)['DeviceIdealSignal'][0:DeviceNum]
    DeviceID = SignalConfig['DeviceID']
    SubframeNum = SignalConfig['SubframeNum']
    validgirdindex = SignalConfig['index']
    IdealSignal = SignalConfig['IdealTimeDomainSignal']
    Sampling_Rate = SignalConfig['Sampling_Rate']
    LongCP = SignalConfig['LongCP']
    ShortCP = SignalConfig['ShortCP']
    Nfft = SignalConfig['Nfft']

    deviceConfigList = generate_device_config(DeviceNum, config)           # 信道参数配置
    subframe_ids = np.unique(SubframeNum)    # 获取所有子帧编号
    for subframe in subframe_ids:
        # 获取该子帧内的所有用户信号
        indices = np.where(SubframeNum == subframe)[0]
        # -------------------------------加噪，加指纹，加信道，模拟接收端的多用户叠加信号----------------------------
        for id in range(len(indices)):
            # 获取当前设备的配置
            currdeviceConfig = deviceConfigList[indices[id]]
            # 加噪
            rx_signal = awgn(IdealSignal[indices[id]], currdeviceConfig['SNRdB']) # 2
            # rx_signal = IdealSignal[indices[id]]
            # 加指纹
            rx_signal = addRFF(Sampling_Rate[indices[id]], rx_signal, DeviceID[indices[id]]) #1
            # 加信道
            rx_signal = addChannel(rx_signal, currdeviceConfig['pathDelays'], currdeviceConfig['avgPathGains'], Sampling_Rate[indices[id]]) #2
            # 子帧内多用户信号叠加
            if(id == 0):
                added_waveform = currdeviceConfig['power'] * rx_signal
            else:
                added_waveform += currdeviceConfig['power'] * rx_signal

        # --------------------------------------解资源格并提取特征-----------------------------------------------
        resourceGrid = showFreqDomain(added_waveform, LongCP[indices[0]], ShortCP[indices[0]], Nfft[indices[0]])
        for i in indices:
            RB_CSI_FFT, RB_CSI = extractData(resourceGrid,validgirdindex[i]-1, DeviceID[i], DeviceID[i]%6)
            if DeviceID[i] / DeviceNum <= 0.6:
                dataset_train = np.vstack((dataset_train, RB_CSI_FFT))   # 垂直拼接
            elif DeviceID[i] / DeviceNum <= 0.8:
                dataset_val = np.vstack((dataset_val, RB_CSI_FFT))
            else:
                dataset_test = np.vstack((dataset_test, RB_CSI_FFT))

    dataset = np.vstack((dataset_train, dataset_val, dataset_test))
    dataset_pd = pd.DataFrame(dataset_train)
    dataset_pd.to_csv(f"{data_dir}/TrainDataset.csv", index=False, header=False)
    dataset_pd = pd.DataFrame(dataset_val)
    dataset_pd.to_csv(f"{data_dir}/ValDataset.csv", index=False, header=False)
    dataset_pd = pd.DataFrame(dataset_test)
    dataset_pd.to_csv(f"{data_dir}/TestDataset.csv", index=False, header=False)
    return dataset[:, :-1], dataset[:, -1]    # 返回数据和标签

def showFreqDomain(data, longCP=320, shortCP=288, Nfft=4096):
    # 提取资源格
    numSymbolsPerSlot = 14
    numSlotsPerSubframe = 1
    numSymbolsPerSubframe = numSymbolsPerSlot * numSlotsPerSubframe
    # 初始化资源格
    resourceGrid = np.zeros((Nfft, numSymbolsPerSubframe), dtype=complex)

    sampleIndex = 0  # 样本索引初始化
    # 处理每个 OFDM 符号
    for symbol in range(numSymbolsPerSubframe):
        # 获取CP长度
        if symbol % numSymbolsPerSlot == 0:
            cpLength = longCP
        else:
            cpLength = shortCP

        # 提取一个完整的OFDM符号（去CP）
        symbolData = data[sampleIndex + cpLength: sampleIndex + cpLength + Nfft]
        sampleIndex += cpLength + Nfft  # 更新样本索引
        # 对每个OFDM符号进行FFT
        freqSymbol = np.fft.fftshift(np.fft.fft(symbolData, Nfft))
        # 将频域符号存入资源格
        resourceGrid[:, symbol] = freqSymbol
    return resourceGrid

# 特征提取：返回时频域CSI
def extractData(resourceGrid,  index,userId, deviceId):
    # 要提取的符号（去DMRS）
    symbolsToExtract = [0, 1, 3, 4, 5, 6]
    # 提取有效数据
    # print("index",len(index))
    data_valid = resourceGrid[index, :]
    data_valid = data_valid[:, symbolsToExtract]
    
    power = np.mean(np.abs(data_valid))
    # print(power)
    # 解调并恢复理想符号（QPSK）
    num_rows, num_cols = data_valid.shape
    demod_bits = np.zeros((num_rows, num_cols, 2), dtype=int)
    # 解调实部和虚部
    demod_bits[:, :, 0] = 2 * (np.real(data_valid) >= 0) - 1
    demod_bits[:, :, 1] = 2 * (np.imag(data_valid) >= 0) - 1
    # 恢复理想符号
    symbol_value = (demod_bits[:, :, 0] + 1j * demod_bits[:, :, 1]) * (1 / np.sqrt(2))

    # 计算CSI
    data_valid = data_valid / symbol_value
    data_valid = data_valid[:, :]  # 删除第一列
    data_valid = np.mean(data_valid, axis=1)  # 计算每一行的均值

    # 每行12个子载波的CSI，对应一个RB
    symbol_data = data_valid.flatten()  # 数据展平
    num_groups = len(symbol_data) // 12 # 计算当前设备使用的RB数量
    reshaped_data = symbol_data[:num_groups * 12].reshape(num_groups, 12)

    power_column = np.full((num_groups, 1), power)

    reshaped_data = np.hstack((reshaped_data, power_column.reshape(num_groups, 1)))
    
    # index_column = np.arange(0, num_groups).reshape(num_groups, 1)+int(index[0]/12)
    # symbol_matrix = np.hstack((reshaped_data, index_column))
    # 添加用户标签列
    device_column = np.full((num_groups, 1), userId)
    symbol_matrix = np.hstack((reshaped_data, device_column))
    # 添加设备标签列
    rff_column = np.full((num_groups, 1), deviceId)
    symbol_matrix = np.hstack((symbol_matrix, rff_column))
    # print(symbol_matrix.shape)
    
    # 对每一行进行FFT
    fft_out = np.fft.fft(symbol_matrix[:, :12], axis=1)
    symbol_matrix[:, :12] = fft_out
    RB_CSI_FFT = symbol_matrix
    real_part = np.real(symbol_matrix[:, :12])
    imag_part = np.imag(symbol_matrix[:, :12])

    # 合并实部和虚部，形成 (100, 24) 的矩阵
    RB_CSI = np.concatenate((real_part, imag_part), axis=1)
    return RB_CSI_FFT, RB_CSI

# 二维可视化
def plot_tsne_2D(data, labels):

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=np.abs(labels), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)

    # 创建颜色映射
    categories = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    # 绘制每个类别的散点图
    for i, category in enumerate(categories):
        plt.scatter(data[labels == category, 0], data[labels == category, 1], label=f'Device {int(category)}', color=colors[i])
    # 添加图例
    plt.legend()

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # plt.savefig( '/workspace/classify/plots/exam.png')
    plt.show()

# 三维可视化
def plot_tsne_3D(data, labels):

    # 创建 3D 绘图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 可视化每个类别的数据点
    for label in np.unique(labels):
        ax.scatter(
            data[labels == label, 0],
            data[labels == label, 1],
            data[labels == label, 2],
            label=f'Device {int(label)}',  # 为每个类别的数据点添加图例标签
            s=50,  # 设置点的大小
            alpha=0.7  # 透明度
        )

    # 添加标题和轴标签
    ax.set_title('3D t-SNE Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')

    # 显示图例
    ax.legend(loc='best')

    # 显示图像
    plt.show()

def main():
    DeviceNum = 60       # 设备数量，默认60
    data_dir = 'D:/Multiuser_detection/generateDataset/dataset'
    path_config = {
        'power': [0.9, 0.9],  # 功率范围 0.5 到 1.0
        'SNR': [30, 30],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 30e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [3, 3]  # 多径数目
    }

    x, y = generate_dataset(data_dir, path_config, DeviceNum)
    # x = abs(x[0:1000,:])
    # y = abs(y[0:1000])
    # tsne = TSNE(n_components=2, random_state=42)
    # data_tsne = tsne.fit_transform(x)
    # plot_tsne_2D(data_tsne, y)

if __name__ == '__main__':
    main()