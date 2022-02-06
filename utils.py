import pickle as pkl

#import networkx as nx
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
# from __future__ import print_function
import numpy as np
import random
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity

def multiclass_node_classification_eval(X, y, ratio=0.2, rnd=2018):
    warnings.filterwarnings('ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=rnd)
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    return macro_f1, micro_f1

def node_classification_F1(Embeddings, y):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        macro_f1, micro_f1 = multiclass_node_classification_eval(Embeddings, y, 0.7, rnd)
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
    macro_f1_avg /= 10
    micro_f1_avg /= 10
    print('Macro_f1 average value: ' + str(macro_f1_avg))
    print('Micro_f1 average value: ' + str(micro_f1_avg))

def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)

def sim(z1, z2, tau):
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t())
    sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)
    return sim_matrix

def evaluate_results_nc(embeddings, labels, num_classes):
    repeat = 20
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)
    print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))

    print('\nK-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes, repeat=repeat)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    macro_mean = [x for (x, y) in svm_macro_f1_list]
    micro_mean = [x for (x, y) in svm_micro_f1_list]
    return np.array(macro_mean), np.array(micro_mean), nmi_mean, ari_mean