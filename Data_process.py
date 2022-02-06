# -*- coding: utf-8 -*-
from collections import defaultdict

import scipy.io as scio
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score

operatorTypes = ["HAD"]
class Load_data():
    def __init__(self, dataset):
        self.dataset = dataset
    def graph_yelp(self):
        H_BL = np.load('./data/Yelp/H_BL.npy')
        H_BP = np.load('./data/Yelp/H_BP.npy')
        H_BS = np.load('./data/Yelp/H_BS.npy')
        location_num = H_BL.shape[0]
        business_num = H_BL.shape[1]
        phrase_num = H_BP.shape[0]
        star_num = H_BS.shape[0]
        return H_BL, H_BP, H_BS, location_num, phrase_num, star_num, business_num
    def graph_pubmed(self):
        H_GC = np.load('./data/PubMed/H_GC.npy')
        H_GD = np.load('./data/PubMed/H_GD.npy')
        H_GS = np.load('./data/PubMed/H_GS.npy')
        chemical_num = H_GC.shape[0]
        gene_num = H_GC.shape[1]
        disease_num = H_GD.shape[0]
        species_num = H_GS.shape[0]
        return H_GC, H_GD, H_GS, chemical_num, disease_num, species_num, gene_num
    def graph_acm(self):
        # 0 for movies, 1 for directors, 2 for actors
        prefix = './data/preprocessed/IMDB_processed'
        adjM = sp.load_npz(prefix + '/adjM.npz')
        type_mask = np.load(prefix + '/node_types.npy')
        adjM = adjM.toarray()
        print(type_mask)
        print(adjM.shape)
        m_mask = np.where(type_mask == 0)[0]
        print(m_mask.shape[0])
        d_mask = np.where(type_mask == 1)[0]
        print(d_mask.shape[0])
        a_mask = np.where(type_mask == 2)[0]
        print(a_mask.shape[0])
        # c_mask = np.where(type_mask == 3)[0]
        # print(c_mask.shape)
        H_V, _ = np.hsplit(adjM, [m_mask.shape[0]])
        _, H_MD, H_MA = np.vsplit(H_V, [m_mask.shape[0], m_mask.shape[0] + d_mask.shape[0]])
        print(H_MD.shape, H_MA.shape)
        labels = np.load(prefix + '/labels.npy')
        print(labels)
        return H_MD, H_MA, m_mask.shape[0], d_mask.shape[0], a_mask.shape[0],labels
    def graph_icdm(self):
        # 0 for movies, 1 for directors, 2 for actors
        prefix = './data/preprocessed/IMDB_processed'
        adjM = sp.load_npz(prefix + '/adjM.npz')
        type_mask = np.load(prefix + '/node_types.npy')
        adjM = adjM.toarray()
        print(type_mask)
        print(adjM.shape)
        m_mask = np.where(type_mask == 0)[0]
        print(m_mask.shape[0])
        d_mask = np.where(type_mask == 1)[0]
        print(d_mask.shape[0])
        a_mask = np.where(type_mask == 2)[0]
        print(a_mask.shape[0])
        # c_mask = np.where(type_mask == 3)[0]
        # print(c_mask.shape)
        H_V, _ = np.hsplit(adjM, [m_mask.shape[0]])
        _, H_MD, H_MA = np.vsplit(H_V, [m_mask.shape[0], m_mask.shape[0] + d_mask.shape[0]])
        print(H_MD.shape, H_MA.shape)
        labels = np.load(prefix + '/labels.npy')
        print(labels)
        return H_MD, H_MA, m_mask.shape[0], d_mask.shape[0], a_mask.shape[0],labels
    def graph_dblp(self):
        prefix = './data/preprocessed/DBLP_processed'
        adjM = sp.load_npz(prefix + '/adjM.npz')
        type_mask = np.load(prefix + '/node_types.npy')
        adjM = adjM.toarray()
        print(type_mask)
        print(adjM.shape)
        a_mask = np.where(type_mask == 0)[0]
        print(a_mask)
        p_mask = np.where(type_mask == 1)[0]
        print(p_mask.shape)
        t_mask = np.where(type_mask == 2)[0]
        print(t_mask.shape)
        c_mask = np.where(type_mask == 3)[0]
        print(c_mask.shape)
        _, H_V, _ = np.hsplit(adjM, [a_mask.shape[0], a_mask.shape[0] + p_mask.shape[0]])
        H_PA, _, H_PT, H_PC = np.vsplit(H_V, [a_mask.shape[0], a_mask.shape[0] + p_mask.shape[0],
                                              a_mask.shape[0] + p_mask.shape[0] + t_mask.shape[0]])
        print(H_PA.shape, H_PT.shape, H_PC.shape)
        labels = np.load(prefix + '/labels.npy')
        return H_PA, H_PC, H_PT, a_mask.shape[0], c_mask.shape[0], t_mask.shape[0], p_mask.shape[0], labels
    def graph1(self):
        author = set()
        paper = set()
        term = set()
        conv = set()
        PA_file = './data/DBLP_four_area/paper_author.txt'
        PC_file = './data/DBLP_four_area/paper_conf.txt'
        PT_file = './data/DBLP_four_area/paper_term.txt'
        with open(PA_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                paper.add(line.split('\t')[0])
                author.add(line.split('\t')[1].strip('\n'))
        with open(PC_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                paper.add(line.split('\t')[0])
                conv.add(line.split('\t')[1].strip('\n'))
        with open(PT_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                paper.add(line.split('\t')[0])
                term.add(line.split('\t')[1].strip('\n'))

        author_dict = {}
        paper_dict = {}
        term_dict = {}
        conf_dict = {}
        num = 0
        for x in author:
            author_dict[x] = num
            num += 1
        num = 0
        for x in paper:
            paper_dict[x] = num
            num += 1
        num = 0
        for x in conv:
            conf_dict[x] = num
            num += 1
        num = 0
        for x in term:
            term_dict[x] = num
            num += 1
        paper_author = []
        paper_conf = []
        paper_term = []
        with open(PT_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_term.append(tup)
                num += 1
            # print(term_dict)
            f.close()

        with open(PC_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_conf.append(tup)
                num += 1
            # print(term_dict)
            f.close()

        with open(PA_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_author.append(tup)
                num += 1
            # print(term_dict)
            f.close()
        print(len(author_dict))
        print(len(paper_dict))
        print(len(term_dict))
        print(len(conf_dict))
        print(len(paper_author))
        print(len(paper_conf))
        print(len(paper_term))
        author_num = len(author_dict)
        paper_num = len(paper_dict)
        term_num = len(term_dict)
        conf_num = len(conf_dict)
        paper_author_edges = len(paper_author)
        paper_term_edges = len(paper_term)
        paper_conf_edges = len(paper_conf)
        H_PA = np.zeros((author_num, paper_num), np.float32)
        for tup in paper_author:
            H_PA[author_dict[tup[1]]][paper_dict[tup[0]]] = 1.
        H_PT = np.zeros((term_num, paper_num), dtype=np.float32)
        for tup in paper_term:
            H_PT[term_dict[tup[1]]][paper_dict[tup[0]]] = 1.
        H_PC = np.zeros((conf_num, paper_num), np.float32)
        for tup in paper_conf:
            H_PC[conf_dict[tup[1]]][paper_dict[tup[0]]] = 1.

        return H_PA, H_PC, H_PT, author_num, conf_num, term_num, paper_num

    def graph(self):
        '''

        :return: 返回三个np邻接矩阵
        '''
        author_dict = {}
        paper_dict = {}
        term_dict = {}
        conf_dict = {}
        author_file = './data/DBLP_four_area/author_label.txt'
        paper_file = './data/DBLP_four_area/paper.txt'
        conf_file = './data/DBLP_four_area/conf.txt'
        term_file = './data/DBLP_four_area/term_modify.txt'
        with open(author_file, 'r',encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                author_dict[line.split('\t')[0]] = num
                num += 1
            # print(author_dict)
            f.close()

        with open(paper_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                paper_dict[line.split('\t')[0]] = num
                num += 1
            # print(paper_dict)
            f.close()

        with open(conf_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                conf_dict[line.split('\t')[0]] = num
                num += 1
            # print(conf_dict)
            f.close()

        with open(term_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                term_dict[line.split('\t')[0]] = num
                num += 1
            # print(term_dict)
            f.close()
        paper_author = []
        paper_conf = []
        paper_term = []
        paper_author_file = './data/DBLP_four_area/paper_author.txt'
        paper_conf_file = './data/DBLP_four_area/paper_conf.txt'
        paper_term_file = './data/DBLP_four_area/paper_term.txt'

        with open(paper_term_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_term.append(tup)
                num += 1
            # print(term_dict)
            f.close()

        with open(paper_conf_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_conf.append(tup)
                num += 1
            # print(term_dict)
            f.close()

        with open(paper_author_file, 'r', encoding='gbk') as f:
            num = 0
            for line in f.readlines():
                tup = (line.split('\t')[0], line.split('\t')[1].strip('\n'))
                paper_author.append(tup)
                num += 1
            # print(term_dict)
            f.close()

        print(len(author_dict))
        print(len(paper_dict))
        print(len(term_dict))
        print(len(conf_dict))
        print(len(paper_author))
        print(len(paper_conf))
        print(len(paper_term))
        author_num = len(author_dict)
        paper_num = len(paper_dict)
        term_num = len(term_dict)
        conf_num = len(conf_dict)
        paper_author_edges = len(paper_author)
        paper_term_edges = len(paper_term)
        paper_conf_edges = len(paper_conf)
        H_PA = np.zeros((author_num, paper_num), np.float32)
        for tup in paper_author:
            H_PA[author_dict[tup[1]]][paper_dict[tup[0]]] = 1.
        H_PT = np.zeros((term_num, paper_num), dtype=np.float32)
        for tup in paper_term:
            H_PT[term_dict[tup[1]]][paper_dict[tup[0]]] = 1.
        H_PC = np.zeros((conf_num, paper_num), np.float32)
        for tup in paper_conf:
            H_PC[conf_dict[tup[1]]][paper_dict[tup[0]]] = 1.

        return H_PA, H_PC, H_PT, author_num, conf_num, term_num, paper_num


class Graph_Construction:
    # Input: n * d.
    def __init__(self, X):
        self.X = X

    def Middle(self):
        Inner_product = self.X.mm(self.X.T)
        Graph_middle = torch.sigmoid(Inner_product)
        return Graph_middle

    # Construct the adjacency matrix by KNN
    def KNN(self, k=9):
        n = self.X.shape[0]
        D = L2_distance_2(self.X, self.X)
        _, idx = torch.sort(D)
        S = torch.zeros(n, n)
        for i in range(n):
            id = torch.LongTensor(idx[i][1: (k + 1)])
            S[i][id] = 1
        S = (S + S.T) / 2
        return S

def Event_matrix(matrix):
    shape = matrix.shape
    result = np.zeros((shape[1], shape[1]), dtype=np.float32)
    for i in range(shape[1]):
        for j in range(shape[1]):
            for k in range(shape[0]):
                if matrix[k][i] == matrix[k][j] and matrix[k][i] == 1:
                    result[i][j] += 1
    return result



def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))


def get_roc_score(edges_pos, edges_neg, emb):
    """ Link Prediction: computes AUC ROC and AP scores from embeddings vectors,
    and from ground-truth lists of positive and negative node pairs
    :param edges_pos: list of positive node pairs
    :param edges_neg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """
    preds = []
    preds_neg = []
    for e in edges_pos:
        # Link Prediction on positive pairs
        preds.append(sigmoid(emb[e[0],e[1]]))
    for e in edges_neg:
        # Link Prediction on negative pairs
        preds_neg.append(sigmoid(emb[e[0], e[1]]))

    # Stack all predictions and labels
    preds_all = np.hstack([preds, preds_neg])

    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    #label = np.ones(len(preds))
    # Computes metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2 * bound * torch.rand(d1, d2)
    return torch.Tensor(nor_W)


def L2_distance_2(A, B):
    A = A.T
    B = B.T
    AA = torch.sum(A * A, dim=0, keepdims=True)
    BB = torch.sum(B * B, dim=0, keepdims=True)
    AB = (A.T).mm(B)
    D = ((AA.T).repeat(1, BB.shape[1])) + (BB.repeat(AA.shape[1], 1)) - 2 * AB
    D = torch.abs(D)
    return D


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, test_percent=20., val_percent=0.01):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    edges_positive, _, shape = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    # edges_positive = edges_positive[edges_positive[:, 1] > edges_positive[:, 0], :]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None
    edge_num = shape[0] * shape[1]
    print('shape', shape)
    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))
    num_test_false = int(edge_num * test_percent / 100. - num_test)
    num_val_false = int(edge_num * val_percent / 100. - num_val)
    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]  # positive test edges
    val_edges = edges_positive[val_edge_idx]  # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0)  # positive train edges
    num_train = train_edges.shape[0]
    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj)  # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:, 0] * adj.shape[1] + positive_idx[:, 1]  # linear indices

    train_edges_false = np.empty((0, 2), dtype='int64')
    idx_train_edges_false = np.empty((0,), dtype='int64')
    while len(train_edges_false) < len(train_edges):
        idx = np.random.choice(adj.shape[0] * adj.shape[1], 2 * (num_train - len(train_edges_false)), replace=True)

        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_train_edges_false, assume_unique=True)]

        rowidx = idx // adj.shape[1]
        colidx = idx % adj.shape[1]
        coords = np.vstack((rowidx, colidx)).transpose()

        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not anymore
        # step 6:
        # coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(int(num_train), len(idx))]
        train_edges_false = np.append(train_edges_false, coords, axis=0)
        idx = idx[:min(int(num_train), len(idx))]
        idx_train_edges_false = np.append(idx_train_edges_false, idx)
        print('sampled', len(train_edges_false))

    test_edges_false = np.empty((0, 2), dtype='int64')
    idx_test_edges_false = np.empty((0,), dtype='int64')
    # todo 查看索引是否对应
    while len(test_edges_false) <  len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0] * adj.shape[1], 2 * (num_test - len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_train_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[1]
        colidx = idx % adj.shape[1]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        #lowertrimask = coords[:, 0] > coords[:, 1]
        #coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not anymore
        # step 6:
        #coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(int(num_test), len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis=0)
        idx = idx[:min(int(num_test), len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)
        print('sampled', len(test_edges_false))
    val_edges_false = np.empty((0, 2), dtype='int64')
    idx_val_edges_false = np.empty((0,), dtype='int64')
    while len(val_edges_false) < num_val:
        # step 1:
        idx = np.random.choice(adj.shape[0] * adj.shape[1], 2 * (num_val - len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_train_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[1]
        colidx = idx % adj.shape[1]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        # lowertrimask = coords[:, 0] > coords[:, 1]
        # coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not any more
        # step 6:
        # coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)
    #
    # sanity checks:
    train_edges_linear = train_edges[:, 0] * adj.shape[1] + train_edges[:, 1]
    test_edges_linear = test_edges[:, 0] * adj.shape[1] + test_edges[:, 1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[1] + val_edges[:, 1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[1] + val_edges[:, 1], test_edges_linear))

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0],dtype=np.float32)
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false, train_edges, train_edges_false

def mask_test_edges_gr(adj, test_percent=20., val_percent=0.01):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    edges_positive, _, shape = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    # edges_positive = edges_positive[edges_positive[:, 1] > edges_positive[:, 0], :]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None
    edge_num = shape[0] * shape[1]
    print('shape', shape)
    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))
    num_test_false = int(edge_num * test_percent / 100. - num_test)
    num_val_false = int(edge_num * val_percent / 100. - num_val)
    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]  # positive test edges
    val_edges = edges_positive[val_edge_idx]  # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0)  # positive train edges
    num_train = train_edges.shape[0]
    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj)  # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:, 0] * adj.shape[1] + positive_idx[:, 1]  # linear indices
    train_edges_false_tuple = np.where(adj == 0)
    train_edges_false_tuple_start = train_edges_false_tuple[0]
    train_edges_false_tuple_des = train_edges_false_tuple[1]
    train_edges_false = np.empty((0, 2), dtype='int64')
    for x, y in train_edges_false_tuple_start, train_edges_false_tuple_des:
        train_edges_false = np.append(train_edges_false, [x, y], axis=0)

    idx_train_edges_false = np.empty((0,), dtype='int64')


    # Re-build adj matrix
    data = np.ones(train_edges.shape[0],dtype=np.float32)
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    return adj_train, train_edges, train_edges_false, train_edges, train_edges_false, train_edges, train_edges_false

def get_link_score(fu, fv, operator):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return np.multiply(fu, fv)
    else:
        raise NotImplementedError

def get_link_feats(links, source_embeddings, target_embeddings, operator):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(source_embeddings[a], target_embeddings[b], operator)
        features.append(f)
    return features

def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])
    test_results_ap = defaultdict(lambda :[])
    # embs = source_embeds.dot(np.transpose(target_embeds))
    #test_auc, test_ap = get_roc_score(test_pos, test_neg, source_embeds.dot(np.transpose(target_embeds)))
    #val_auc = get_roc_score(val_pos, val_neg, source_embeds.dot(np.transpose(target_embeds)))

    #test_results['SIGMOID'].extend([test_auc, test_auc])
    #val_results['SIGMOID'].extend([val_auc, val_auc])
    #test_results_ap['SIGMOID'].extend([test_ap, test_ap])
    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])
    for operator in operatorTypes:
        train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds, operator))
        train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds, operator))
        val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds, operator))
        val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds, operator))
        test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds, operator))
        test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds, operator))

        train_pos_labels = np.array([1] * len(train_pos_feats))
        train_neg_labels = np.array([-1] * len(train_neg_feats))
        val_pos_labels = np.array([1] * len(val_pos_feats))
        val_neg_labels = np.array([-1] * len(val_neg_feats))

        test_pos_labels = np.array([1] * len(test_pos_feats))
        test_neg_labels = np.array([-1] * len(test_neg_feats))
        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.append(train_pos_labels, train_neg_labels)

        val_data = np.vstack((val_pos_feats, val_neg_feats))
        val_labels = np.append(val_pos_labels, val_neg_labels)

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.append(test_pos_labels, test_neg_labels)

        logistic = linear_model.LogisticRegression()
        logistic.fit(train_data, train_labels)
        test_predict = logistic.predict_proba(test_data)[:, 1]
        val_predict = logistic.predict_proba(val_data)[:, 1]

        test_roc_score = roc_auc_score(test_labels, test_predict)
        val_roc_score = roc_auc_score(val_labels, val_predict)
        test_ap_score = average_precision_score(test_labels, test_predict)
        val_results[operator].extend([val_roc_score, val_roc_score])
        test_results[operator].extend([test_roc_score, test_roc_score])
        test_results_ap[operator].extend([test_ap_score, test_ap_score])
        val_pred_true[operator].extend(zip(val_predict, val_labels))
        test_pred_true[operator].extend(zip(test_predict, test_labels))

    return val_results, test_results,test_results_ap, val_pred_true, test_pred_true