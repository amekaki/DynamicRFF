import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def torch_distance(x):
    x_x = (x * x).sum(1).repeat(1, x.size(0))
    x2_x = x.mm(x.t())
    return x_x + x_x.t() - 2 * x2_x


def torch_distance_mask(label):
    label = label.view(len(label), 1)
    label = label.repeat(1, label.size(0))
    index = torch.arange(0, label.size(0)).repeat(label.size(0), 1).to(label.device)
    intra_mask = ((label.eq(label.t())) + (index > index.t())).eq(2)
    inter_mask = ((label.ne(label.t())) + (index > index.t())).eq(2)
    return inter_mask, intra_mask


def inter_intra_dist(features, labels, metric='cosine', other_features=None):
    if not other_features:
        label_repeat = np.tile(labels, (len(labels), 1))
        itself_dist = np.eye(len(labels))
        intra_dist_mask = ((label_repeat == label_repeat.T) -
                           itself_dist).astype(np.bool)
        inter_dist_mask = label_repeat != label_repeat.T

        distance = pairwise_distances(features, metric=metric)

        intra_dist = distance[intra_dist_mask]
        inter_dist = distance[inter_dist_mask]

    else:
        distance = pairwise_distances(features, other_features, metric=metric)
        pair_dist = np.diag(distance)
        # print(features.shape)
        # print(other_features.shape)
        # print(pair_dist.shape)
        inter_dist = pair_dist[labels == 0]
        intra_dist = pair_dist[labels != 0]

    return intra_dist, inter_dist

def pair_dist(features1, features2, metric='cosine'):
    distance = pairwise_distances(features1, features2, metric=metric)
    pair_dist = np.diag(distance)
    return pair_dist

def compute_centers(features, labels):
    class_num = len(list(set(labels)))
    N = features.shape[0]
    D = features.shape[1]
    centers = np.zeros((class_num, D))
    center_count = np.zeros_like(centers)
    for i in range(len(labels)):
        index = labels[i]
        centers[index] += features[i]
        center_count[index] += np.ones_like(features[i])

    centers /= center_count
    return centers


def center_dist(features, labels, centers):
    class_num = len(centers)
    N = features.shape[0]
    D = features.shape[1]
    mask = np.zeros((N, class_num))
    mask[np.arange(N), labels] = 1

    distance = pairwise_distances(features, centers)
    intra_dist = distance[mask == 1]
    inter_dist = distance[mask == 0]
    return intra_dist, inter_dist

def distance_hist_plot(intra_dist, inter_dist, filename=None):
    plt.figure(figsize=(4,3), dpi=300)
    # plt.title('Distance distribution')
    plt.hist(intra_dist, 100, density=True, alpha=0.5)
    plt.hist(inter_dist, 100, density=True, alpha=0.5)
    plt.legend(['Intra dist', 'Inter dist'])
    plt.xlabel('Distance')
    plt.ylabel('Density')
    if not filename is None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    


def get_auc_eer(intra_dist, inter_dist, plot_roc=False, filename=None):
    inter_label = np.ones_like(inter_dist)
    intra_label = np.zeros_like(intra_dist)
    y_test = np.append(inter_label, intra_label)
    y_score = np.append(inter_dist, intra_dist)
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)
    if plot_roc:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        if not filename is None:
            plt.savefig(filename)
            plt.close()

    return eer, roc_auc, thresh


def cos_clf(feature_x, feature_t, threshold=0.79):
    cosine_sim = -F.cosine_similarity(feature_x, feature_t).view(-1, 1) + 1
    output = torch.cat((-cosine_sim + 2*threshold, cosine_sim), dim=1)
    return output

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output, norm

def BO_balance_score(RB, auc, lamda=20000):
    return RB - (1-auc)* lamda