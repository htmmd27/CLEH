import sys
from model import *
from Data_process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
import warnings
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import utils
warnings.filterwarnings('ignore')
Dataset = 'dblp'
Link_Prediction = True

Epoch_num = 200
Learning_Rate = 0.001
hidden_dim = 128
hidden_dim1 = 512
hidden_dim2 = 512
hidden_dim3 = 256
d_out = 128
flow_layers = 4
device = torch.device('cuda')
tau = 0.5
def pos(Event_Graph):
    # shape = Event_Graph.shape[0]
    Event_Graph[Event_Graph > 0] = 1
    print(min(Event_Graph))
    return Event_Graph

load_data = Load_data(Dataset)
H_PA, H_PC, H_PT, author_num, conf_num, term_num, paper_num, labels = load_data.graph_dblp()
# 构造训练矩阵 默认80%训练集
#todo 分别取训练集
H_V = np.vstack((H_PA, H_PC))
H_V = np.vstack((H_V, H_PT))
#adj_train, val_edges, val_edges_false, test_edges, test_edges_false, train_edges, train_edges_false = mask_test_edges(sp.coo_matrix(H_V))

adj_a_train, val_a_edges, val_a_edges_false, test_a_edges, test_a_edges_false, train_a_edges, train_a_edges_false = mask_test_edges(sp.coo_matrix(H_PA))
adj_c_train, val_c_edges, val_c_edges_false, test_c_edges, test_c_edges_false, train_c_edges, train_c_edges_false = mask_test_edges(sp.coo_matrix(H_PC))
adj_t_train, val_t_edges, val_t_edges_false, test_t_edges, test_t_edges_false, train_t_edges, train_t_edges_false = mask_test_edges(sp.coo_matrix(H_PT))

adj_a_train = adj_a_train.todense()
adj_c_train = adj_c_train.todense()
adj_t_train = adj_t_train.todense()
adj_train = np.vstack((adj_a_train, adj_c_train))
adj_train = np.vstack((adj_train, adj_t_train))

for i in train_c_edges:
    i[0] += author_num
for i in train_c_edges_false:
    i[0] += author_num
for i in train_t_edges:
    i[0] += author_num
    i[0] += conf_num
for i in train_t_edges_false:
    i[0] += author_num
    i[0] += conf_num

for i in val_c_edges:
    i[0] += author_num
for i in val_c_edges_false:
    i[0] += author_num
for i in val_t_edges:
    i[0] += author_num
    i[0] += conf_num
for i in val_t_edges_false:
    i[0] += author_num
    i[0] += conf_num

for i in test_c_edges:
    i[0] += author_num
for i in test_c_edges_false:
    i[0] += author_num
for i in test_t_edges:
    i[0] += author_num
    i[0] += conf_num
for i in test_t_edges_false:
    i[0] += author_num
    i[0] += conf_num

train_edges = np.vstack((train_a_edges, train_c_edges))
train_edges = np.vstack((train_edges, train_t_edges))
train_edges_false = np.vstack((train_a_edges_false, train_c_edges_false))
train_edges_false = np.vstack((train_edges_false, train_t_edges_false))

val_edges = np.vstack((val_a_edges, val_c_edges))
val_edges = np.vstack((val_edges, val_t_edges))
val_edges_false = np.vstack((val_a_edges_false, val_c_edges_false))
val_edges_false = np.vstack((val_edges_false, val_t_edges_false))

test_edges = np.vstack((test_a_edges, test_c_edges))
test_edges = np.vstack((test_edges, test_t_edges))
test_edges_false = np.vstack((test_a_edges_false, test_c_edges_false))
test_edges_false = np.vstack((test_edges_false, test_t_edges_false))


print(len(train_edges))
for i in range(len(train_edges)):
    if H_V[train_edges[i][0]][train_edges[i][1]] != 1.0:
        print(False)
        break
print(len(test_edges))
for i in range(len(test_edges)):
    if H_V[test_edges[i][0]][test_edges[i][1]] != 1.0:
        print(False)
        break
print(len(test_edges_false))
for i in range(len(test_edges_false)):
    if H_V[test_edges_false[i][0]][test_edges_false[i][1]] != 0.0:
        print(False)
        break
print(H_V[test_edges_false[0][0]][test_edges_false[0][1]])
H_V_train = adj_train
print(H_V_train)
H_PA_train, H_PC_train, H_PT_train = np.vsplit(H_V_train, [author_num, author_num + conf_num])
# 训练集
H_EA_train = torch.from_numpy(H_PA_train)
H_EC_train = torch.from_numpy(H_PC_train)
H_ET_train = torch.from_numpy(H_PT_train)
H_EP_train = torch.eye(paper_num, dtype=torch.float32, device=device)
H_E_train = torch.from_numpy(H_V_train)

H_EAE_train = np.dot(H_PA_train.T, H_PA_train)
row, col = np.diag_indices_from(H_EAE_train)
H_EAE_train[row, col] = 1
pos_EAE = pos(H_EAE_train)
H_EAE_train = torch.from_numpy(H_EAE_train)
pos_EAE = torch.from_numpy(pos_EAE)

H_ETE_train = np.dot(H_PT_train.T, H_PT_train)
row, col = np.diag_indices_from(H_ETE_train)
H_ETE_train[row, col] = 1
pos_ETE = pos(H_ETE_train)
H_ETE_train = torch.from_numpy(H_ETE_train)
pos_ETE = torch.from_numpy(pos_ETE)

H_ECE_train = np.dot(H_PC_train.T, H_PC_train)
row, col = np.diag_indices_from(H_ECE_train)
H_ECE_train[row, col] = 1
pos_ECE = pos(H_ECE_train)
H_ECE_train = torch.from_numpy(H_ECE_train)
pos_ECE = torch.from_numpy(pos_ECE)


# H_EA_train = H_EA_train.to(device)
# H_EC_train = H_EC_train.to(device)
# H_ET_train = H_ET_train.to(device)
# H_EAE_train = H_EAE_train.to(device)
# H_ETE_train = H_ETE_train.to(device)
# H_ECE_train = H_ECE_train.to(device)
# pos_EAE = pos_EAE.to(device)
# pos_ETE = pos_ETE.to(device)
# pos_ECE = pos_ECE.to(device)
# 初始化结果
roc_score = []
ap_score = []
svm_macro_avg = np.zeros((7, ), dtype=np.float)
svm_micro_avg = np.zeros((7, ), dtype=np.float)
micro_f1 = []
macro_f1 = []
nmi = []
ari = []
# 定义loss函数

def loss_function(pos_EAE, pos_ETE, pos_ECE, latent_EA, latent_EC, latent_ET, tau, u_A, u_C, u_T, u_P, log_det_jacobianA, log_det_jacobianC, log_det_jacobianP, log_det_jacobianT):
    matrix_EAE = utils.sim(latent_EA, latent_EA, tau)
    matrix_ECE = utils.sim(latent_EC, latent_EC, tau)
    matrix_ETE = utils.sim(latent_ET, latent_ET, tau)

    matrix_EAE = matrix_EAE / (torch.sum(matrix_EAE, dim=1).view(-1, 1) + 1e-8)
    lori_EAE = -torch.log(matrix_EAE.mul(pos_EAE).sum(dim=-1)).mean()

    matrix_ECE = matrix_ECE / (torch.sum(matrix_ECE, dim=1).view(-1, 1) + 1e-8)
    lori_ECE = -torch.log(matrix_ECE.mul(pos_ECE).sum(dim=-1)).mean()

    matrix_ETE = matrix_ETE / (torch.sum(matrix_ETE, dim=1).view(-1, 1) + 1e-8)
    lori_ETE = -torch.log(matrix_ETE.mul(pos_ETE).sum(dim=-1)).mean()
    lori = lori_EAE + lori_ECE + lori_ETE
    z = MultivariateNormal(torch.zeros(hidden_dim, device=device), torch.eye(hidden_dim,device=device))
    z_loss_A = -1 * torch.mean(z.log_prob(u_A) + torch.mean(log_det_jacobianA, dim=1))
    z_loss_C = -1 * torch.mean(z.log_prob(u_C) + torch.mean(log_det_jacobianC, dim=1))

    z_loss_P = -1 * torch.mean(z.log_prob(u_P) + torch.mean(log_det_jacobianP, dim=1))

    z_loss_T = -1 * torch.mean(z.log_prob(u_T) + torch.mean(log_det_jacobianT, dim=1))

    z_loss = z_loss_A + z_loss_C + z_loss_P + z_loss_T

    return  lori + z_loss

model = myModel_DBLP(author_num, conf_num, paper_num, term_num, paper_num, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out, flow_layers)
optimzer = torch.optim.Adam(model.parameters(), lr = Learning_Rate)
start_time = time.time()
# model.to(device)

for epoch in range(Epoch_num):
    EventA_embedding, EventC_embedding, EventT_embedding, reconstructA, reconstructC, reconstructT, u_A, u_C, u_P, u_T, predicted_A, predicted_C, predicted_P, predicted_T, log_det_jacobianA, log_det_jacobianC, log_det_jacobianP, log_det_jacobianT, H_mean_A, H_mean_C, H_mean_P, H_mean_T = model(H_EA_train, H_EC_train, H_EP_train, H_ET_train)
    loss = loss_function(pos_EAE, pos_ETE, pos_ECE, EventA_embedding, EventC_embedding, EventT_embedding, tau, u_A, u_C, u_T, u_P, log_det_jacobianA, log_det_jacobianC, log_det_jacobianP, log_det_jacobianT)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    LatentPresentation_A = H_mean_A.detach().numpy()
    LatentPresentation_C = H_mean_C.detach().numpy()
    LatentPresentation_T = H_mean_T.detach().numpy()
    LatentPresentation = np.vstack((LatentPresentation_A, LatentPresentation_C))
    LatentPresentation = np.vstack((LatentPresentation, LatentPresentation_T))
    LatentPresentation_P = H_mean_P.detach().numpy()
    LatentPresentation_final = LatentPresentation.dot(np.transpose(LatentPresentation_P))
    val_results, test_results, test_results_ap, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges,
                                                                           val_edges_false, test_edges,
                                                                           test_edges_false, LatentPresentation,
                                                                           LatentPresentation_P)
    print("Epoch: [{}]/[{}]".format(epoch + 1, Epoch_num))
    # print("AUC = {}".format(roc_score_temp))
    # print("AP = {}".format(ap_score_temp))
    epoch_auc_val = val_results["HAD"][1]
    epoch_auc_test = test_results["HAD"][1]
    roc_score.append(epoch_auc_test)
    epoch_ap_test = test_results_ap["HAD"][1]
    ap_score.append(epoch_ap_test)

    #print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
    print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
    print("Epoch {}, Test AP {}".format(epoch, epoch_ap_test))

print("max roc: ", max(roc_score))
print("max ap: ", max(ap_score))

