from trainer_rff_classify import Trainer as rffTrainer
from trainer_user import Trainer as userTrainer
from src.datasetv4 import RFdataset
from src.evaluation import *
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from collections import Counter
from scipy.stats import mode
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels, filename, figsize=(10, 8), dpi=300):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title("Confusion Matrix", fontsize=16)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

# 假设 RB_feature 是 RB 特征矩阵，user_label 是目标标签，RB 是特征数据，threshold 是距离阈值
def purity_score(y_true, y_pred):
    clusters = np.unique(y_pred)
    total_samples = len(y_true)
    purity = 0.0

    for cluster in clusters:
        cluster_indices = np.where(y_pred == cluster)[0]
        true_labels_in_cluster = y_true[cluster_indices]
        
        most_common_label_count = Counter(true_labels_in_cluster).most_common(1)[0][1]
        purity += most_common_label_count
    
    return purity / total_samples

def get_clusters(RB_feature, threshold):
    """
    逐个计算距离并根据阈值合并组。
    """
    n_samples = len(RB_feature)
    clusters = []
    current_cluster = [0]  # 第一个RB作为初始组
    for i in range(1, n_samples):
        # print(i,current_cluster[-1])
        dist = pairwise_distances(RB_feature[i].reshape(1, -1), RB_feature[i-1].reshape(1, -1), metric='cosine')[0, 0]
        # print(dist)
        if dist <= threshold:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    clusters.append(current_cluster) 
    return clusters

def method_2(RB_feature, threshold):
    """
    方法 2：直接根据阈值进行聚类。
    """
    dist_matrix = pairwise_distances(RB_feature, metric='cosine')
    clusters = np.where(dist_matrix <= threshold, 1, 0)  # 阈值小于threshold的视为同一类
    print(clusters.shape)
    return clusters

# 计算 t-SNE 图
def plot_tsne(RB_feature, clusters, title):
    tsne = TSNE(n_components=2, random_state=42)
    RB_feature_tsne = tsne.fit_transform(RB_feature)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(RB_feature_tsne[:, 0], RB_feature_tsne[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.savefig('tsne.png', bbox_inches='tight')

def check(features,labels,trainer,eval_dataset='pipline'):
  intra_dist, inter_dist = inter_intra_dist(features.cpu().detach().numpy(), labels.cpu().numpy())
  distance_hist_plot(intra_dist, inter_dist, filename='./plots/test_{}_{}_dist_hist.png'.format(trainer.flag, eval_dataset))
  eer, roc_auc, thresh = get_auc_eer(intra_dist, inter_dist, plot_roc=True, filename='./plots/test_{}_{}_roc.png'.format(trainer.flag, eval_dataset))
  print(eer, roc_auc, thresh)
  return thresh
def show_rff(x,y):
    y_unique = np.unique(y)
    for label in y_unique:
        x_device =np.real(x[y==label][:,1:13]) 
        print(x_device.shape)
        mean_values = np.mean(x_device, axis=0)
        plt.plot(mean_values, label=f"User {label}")
    plt.title("Mean Line Plot of x_device for Each User")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(alpha=0.7, linestyle="--")
    plt.tight_layout()
    plt.savefig("freature_mean.png")

if __name__ == '__main__':
    import marveltoolbox as mt 
    config = {
        'power': [0.5, 1.0],  # 功率范围 0.5 到 1.0
        'SNR': [25, 35],  # 信噪比范围 25 到 35，单位dB
        'PathDelay': [0, 30e-6],  # 路径延迟范围
        'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
        'PathNum': [2, 5]  # 多径数量
    }
    # UserNum: 用户数量，默认60，按6:2:2划分
    # regenerate_flag: 生成数据集标志
    rff_trainer = rffTrainer(device=1, z_dim=64, config=config, ChannelNum=6, DeviceNum=10, regenerate_flag=False, label='rff')
    rff_trainer.run(load_best=True, retrain=False, is_del_loger=False)
    params = mt.utils.params_count(rff_trainer.models['C'])
    rff_model = rff_trainer.models['C']
    print("rff model test")
    rff_trainer.eval(0,'test')
    user_trainer = userTrainer(device=1, z_dim=64, config=config, ChannelNum=6, DeviceNum=10, regenerate_flag=False, label='user')
    user_trainer.run(load_best=True, retrain=False, is_del_loger=False)
    params = mt.utils.params_count(user_trainer.models['C'])
    user_model = user_trainer.models['C']
    user_trainer.eval(0, 'val')
    test_data = user_trainer.datasets['close']
    config = {
    'power': [0.7, 1.0],  # 功率范围 0.5 到 1.0
    'SNR': [15, 30],  # 信噪比范围 25 到 35，单位dB
    'PathDelay': [0, 5e-5],  # 路径延迟范围
    'AvgPathGain': [-30, -20],  # 路径增益范围，单位dB
    'PathNum': [2, 8]  # 多径数量
    }
    test_data = RFdataset(config=config, ChannelNum=3, DeviceNum=10, regenerate_flag=False, label='rff', datasetname='test')
    # test_data.random_channel(3, 10, config)
    RB_data = test_data.data['x'].to(user_trainer.device)
    print(RB_data.shape)
    rff_label = test_data.data['y_rff']
    user_label = test_data.data['y_user']
    RB_feature = user_model.features(RB_data)
    threshold = check(RB_feature,user_label,user_trainer)
    RB_feature = RB_feature.cpu().detach().numpy()
    clusters_method_1 = get_clusters(RB_feature, threshold)
    user_name = np.unique(user_label)

    # 计算聚类结果纯度
    pure = 0
    for clu in clusters_method_1:
        clu_true_label = user_label[clu]
        counts = np.bincount(clu_true_label)
        most_frequent_count = counts.max()
        most_frequent_percentage = most_frequent_count / len(clu_true_label) * 100
        pure+=most_frequent_percentage
    pure = pure/len(clusters_method_1)
    clusters = clusters_method_1
    # rff_feature = rff_model.features(RB_data)
    scores = rff_model(RB_data)
    pred_rff_label = torch.argmax(scores, dim=1).cpu().numpy()
    rff_label = rff_label.numpy()
    correct = np.sum(pred_rff_label == rff_label)
    print(correct/len(pred_rff_label))
    plot_confusion_matrix(rff_label, pred_rff_label, labels=np.unique(rff_label), filename='signle')

    scores = rff_model(RB_data)
    pred_rff_label = torch.argmax(scores, dim=1).cpu().numpy()
    clusters = get_clusters(RB_feature, threshold)
    for cluster_index in clusters:
        cluster_labels = pred_rff_label[cluster_index]
        values, counts = np.unique(cluster_labels, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        pred_rff_label[cluster_index] = most_frequent_value
    correct = np.sum(pred_rff_label == rff_label)


    plot_confusion_matrix(rff_label, pred_rff_label, labels=np.unique(rff_label), filename='vote')

    print(correct/len(pred_rff_label))

    # purity_score = 0
    # for cluster in clusters:
    #     print(len(cluster))
    #     cluster_true_label = user_label[cluster]
    #     counts = np.bincount(cluster_true_label)
    #     most_frequent_count = counts.max()
    #     most_frequent_percentage = most_frequent_count / len(cluster_true_label) * 100
    #     purity_score+=most_frequent_percentage
    # purity_score = purity_score/len(clusters_method_1)

    # print(pure)
    # clusters_method_2 = method_2(RB_feature, threshold)
    # print(len(clusters_method_1),len(clusters_method_2))
    # 计算纯度
    # print(len(clusters_method_1),threshold)
    # print(len(user_name))
    
    # purity_method_1 = purity_score(user_label, np.array([label for cluster in clusters_method_1 for label in [user_label[cluster[0]]] * len(cluster)]))
    # purity_method_2 = purity_score(user_label, clusters_method_2.flatten())

    # print(f"方法 1 的纯度: {purity_method_1:.4f}")
    # print(f"方法 2 的纯度: {purity_method_2:.4f}")

    # 可视化 t-SNE 图
    # plot_tsne(RB_feature, np.array([label for cluster in clusters_method_1 for label in [user_label[cluster[0]]] * len(cluster)]), title="Method 1 t-SNE")
    # plot_tsne(RB_feature, clusters_method_2.flatten(), title="Method 2 t-SNE")
    # print()
    # for frame_index in frame_index_list:
    #    RB = RB_data[frame_label == frame_index]
    #    user_target = user_label[frame_label == frame_index]
    #    RB_feature = user_model.features(RB)
          
      #  print(RB_rff.shape)
    # print(RB_feature.shape)
    
    # check(RB_rff,rff_label,rff_trainer)
    # check(RB_feature,user_label,user_trainer)
    
    # user_trainer.eval(0,ext_name='_test_check')