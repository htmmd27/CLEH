import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn

from Data_process import get_weight_initial
from layers import RealNVP

device = torch.device('cuda')

class myModel_DBLP(nn.Module):
    def __init__(self, d_inA, d_inC, d_inP, d_inT, d_inEvents, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out, flow_layers):
        '''

        :param d_inA: 作者个数
        :param d_inC: 会议个数
        :param d_inP: 论文个数
        :param d_inT: 关键词个数
        :param d_inEvents: 超边个数
        :param hidden_dim1: ecnoder维度
        :param hidden_dim2: 流模型维度
        :param flow_layers: ；流模型层数
        :param hidden_dim3: events重构隐藏层
        :param d_out: 最终输出维度
        '''
        super(myModel_DBLP, self).__init__()

        # 初始encoder 使用mlp 激活函数使用tanh
        self.mlpA = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpC = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpP = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpT = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )
        # self.mlpA_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpC_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpP_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpT_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # Trans
        self.transA = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transC = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transP = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transT = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        # 定义流模型
        self.flowA = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowC = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowP = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowT = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # events重构 mlp,一共k - 1个
        self.mlpEventsA = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsC = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsT = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )

    def Encoder(self, H_A, H_C, H_P, H_T):
        H_mean_A = self.mlpA(H_A.mm(torch.eye(H_A.size(1), dtype=torch.float32)))
        H_mean_C = self.mlpC(H_C.mm(torch.eye(H_C.size(1), dtype=torch.float32)))
        H_mean_P = self.mlpP(H_P.mm(torch.eye(H_P.size(1), dtype=torch.float32)))
        H_mean_T = self.mlpT(H_T.mm(torch.eye(H_T.size(1), dtype=torch.float32)))
        H_mean_A = self.transA(H_mean_A)
        H_mean_C = self.transC(H_mean_C)
        H_mean_P = self.transA(H_mean_P)
        H_mean_T = self.transC(H_mean_T)
        return H_mean_A, H_mean_C, H_mean_P, H_mean_T
    def Pooling(self, H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T):
        shape_A = H_A.shape[1]
        shape_C = H_C.shape[1]
        shape_T = H_T.shape[1]
        listA = [1] * shape_A
        listC = [1] * shape_C
        listT = [1] * shape_T
        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)
        sum_list_C = []
        for tensor in torch.split(H_C, listC, dim=1):
            sum_pooling = torch.sum((H_mean_C * tensor), dim=0)
            sum_list_C.append(sum_pooling)
        sum_C = torch.stack(sum_list_C, dim=0)
        sum_list_T = []
        for tensor in torch.split(H_T, listT, dim=1):
            sum_pooling = torch.sum((H_mean_T * tensor), dim=0)
            sum_list_T.append(sum_pooling)
        sum_T = torch.stack(sum_list_T, dim=0)
        EventA_embedding = torch.cat((H_mean_P, sum_A), dim=1)
        EventC_embedding = torch.cat((H_mean_P, sum_C), dim=1)
        EventT_embedding = torch.cat((H_mean_P, sum_T), dim=1)
        return EventA_embedding, EventC_embedding, EventT_embedding

    def forward(self, H_A, H_C, H_P, H_T):
        H_mean_A, H_mean_C, H_mean_P, H_mean_T = self.Encoder(H_A, H_C, H_P, H_T)
        u_A, log_det_jacobianA = self.flowA(H_mean_A)
        u_C, log_det_jacobianC = self.flowC(H_mean_C)
        u_P, log_det_jacobianP = self.flowP(H_mean_P)
        u_T, log_det_jacobianT = self.flowT(H_mean_T)
        predicted_A = self.flowA.inverse(u_A)
        predicted_C = self.flowC.inverse(u_C)
        predicted_P = self.flowA.inverse(u_P)
        predicted_T = self.flowT.inverse(u_T)
        EventA_embedding, EventC_embedding, EventT_embedding = self.Pooling(H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T)
        reconstructA = self.mlpEventsA(EventA_embedding)
        reconstructC = self.mlpEventsC(EventC_embedding)
        reconstructT = self.mlpEventsT(EventT_embedding)
        return EventA_embedding, EventC_embedding, EventT_embedding, reconstructA, reconstructC, reconstructT, u_A, u_C, u_P, u_T, predicted_A, predicted_C, predicted_P, predicted_T, log_det_jacobianA, log_det_jacobianC, log_det_jacobianP, log_det_jacobianT, H_mean_A, H_mean_C, H_mean_P, H_mean_T

class myModel_DBLP_1(nn.Module):
    def __init__(self, d_inA, d_inC, d_inP, d_inT, d_inEvents, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out, flow_layers):
        '''

        :param d_inA: 作者个数
        :param d_inC: 会议个数
        :param d_inP: 论文个数
        :param d_inT: 关键词个数
        :param d_inEvents: 超边个数
        :param hidden_dim1: ecnoder维度
        :param hidden_dim2: 流模型维度
        :param flow_layers: ；流模型层数
        :param hidden_dim3: events重构隐藏层
        :param d_out: 最终输出维度
        '''
        super(myModel_DBLP_1, self).__init__()

        # 初始encoder 使用mlp 激活函数使用tanh
        self.mlpA = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpC = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpP = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpT = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )
        # self.mlpA_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpC_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpP_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpT_mean = torch.nn.Linear(hidden_dim1, hidden_dim)

        # 定义流模型
        self.flowA = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowC = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowP = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        self.flowT = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # events重构 mlp,一共k - 1个
        self.mlpEventsA = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsC = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsT = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )

    def Encoder(self, H_A, H_C, H_P, H_T):
        H_mean_A = self.mlpA(H_A.mm(torch.eye(H_A.size(1), dtype=torch.float32)))
        H_mean_C = self.mlpC(H_C.mm(torch.eye(H_C.size(1), dtype=torch.float32)))
        H_mean_P = self.mlpP(H_P.mm(torch.eye(H_P.size(1), dtype=torch.float32)))
        H_mean_T = self.mlpT(H_T.mm(torch.eye(H_T.size(1), dtype=torch.float32)))
        return H_mean_A, H_mean_C, H_mean_P, H_mean_T
    def Pooling(self, H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T):
        shape_A = H_A.shape[1]
        shape_C = H_C.shape[1]
        shape_T = H_T.shape[1]
        listA = [1] * shape_A
        listC = [1] * shape_C
        listT = [1] * shape_T
        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)
        sum_list_C = []
        for tensor in torch.split(H_C, listC, dim=1):
            sum_pooling = torch.sum((H_mean_C * tensor), dim=0)
            sum_list_C.append(sum_pooling)
        sum_C = torch.stack(sum_list_C, dim=0)
        sum_list_T = []
        for tensor in torch.split(H_T, listT, dim=1):
            sum_pooling = torch.sum((H_mean_T * tensor), dim=0)
            sum_list_T.append(sum_pooling)
        sum_T = torch.stack(sum_list_T, dim=0)
        EventA_embedding = torch.cat((H_mean_P, sum_A), dim=1)
        EventC_embedding = torch.cat((H_mean_P, sum_C), dim=1)
        EventT_embedding = torch.cat((H_mean_P, sum_T), dim=1)
        return EventA_embedding, EventC_embedding, EventT_embedding

    def forward(self, H_A, H_C, H_P, H_T):
        H_mean_A, H_mean_C, H_mean_P, H_mean_T = self.Encoder(H_A, H_C, H_P, H_T)
        u_A, log_det_jacobianA = self.flowA(H_mean_A)
        u_C, log_det_jacobianC = self.flowC(H_mean_C)
        u_P, log_det_jacobianP = self.flowP(H_mean_P)
        u_T, log_det_jacobianT = self.flowT(H_mean_T)
        z = MultivariateNormal(torch.zeros(u_A.shape[1]), torch.eye(u_A.shape[1]))
        new_sampled_A = z.sample((H_A.shape[0],))
        predicted_A = self.flowA.inverse(new_sampled_A)
        new_sampled_C = z.sample((H_C.shape[0],))
        predicted_C = self.flowC.inverse(new_sampled_C)
        new_sampled_P = z.sample((H_P.shape[0],))
        predicted_P = self.flowA.inverse(new_sampled_P)
        new_sampled_T = z.sample((H_T.shape[0],))
        predicted_T = self.flowT.inverse(new_sampled_T)
        EventA_embedding, EventC_embedding, EventT_embedding = self.Pooling(H_A, H_C, H_T, new_sampled_A, new_sampled_C, new_sampled_P, new_sampled_T)
        reconstructA = self.mlpEventsA(EventA_embedding)
        reconstructC = self.mlpEventsC(EventC_embedding)
        reconstructT = self.mlpEventsT(EventT_embedding)
        return reconstructA, reconstructC, reconstructT, u_A, u_C, u_P, u_T, predicted_A, predicted_C, predicted_P, predicted_T, log_det_jacobianA, log_det_jacobianC, log_det_jacobianP, log_det_jacobianT, new_sampled_A, new_sampled_C, new_sampled_P, new_sampled_T

class myModel_DBLP_vae(nn.Module):
    def __init__(self, d_inA, d_inC, d_inP, d_inT, d_inEvents, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out):
        super(myModel_DBLP_vae, self).__init__()
        # A Encoder
        self.mlpA = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh())
        self.mlpA[0].weight.data = get_weight_initial(hidden_dim1, d_inEvents)
        self.mlpA_mean = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpA_mean.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)

        self.mlpA_std = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpA_std.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)
        # C Encoder
        self.mlpC = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh())
        self.mlpC[0].weight.data = get_weight_initial(hidden_dim1, d_inEvents)
        self.mlpC_mean = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpC_mean.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)

        self.mlpC_std = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpC_std.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)
        # P Encoder
        self.mlpP = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh())
        self.mlpP[0].weight.data = get_weight_initial(hidden_dim1, d_inEvents)
        self.mlpP_mean = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpP_mean.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)

        self.mlpP_std = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpP_std.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)
        # T Encoder
        self.mlpT = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.Tanh())
        self.mlpT[0].weight.data = get_weight_initial(hidden_dim1, d_inEvents)
        self.mlpT_mean = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpT_mean.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)

        self.mlpT_std = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.mlpT_std.weight.data = get_weight_initial(hidden_dim2, hidden_dim1)
        #Trans
        self.transA = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transC = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transP = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transT = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        #Decoder
        self.mlpEventsA = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim2, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsC = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim2, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsT = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim2, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
    def Encoder(self, H_A, H_C, H_P, H_T):
        H_out_A = self.mlpA(H_A.mm(torch.eye(H_A.size(1), dtype=torch.float32, device=device)))
        H_out_C = self.mlpC(H_C.mm(torch.eye(H_C.size(1), dtype=torch.float32, device=device)))
        H_out_P = self.mlpP(H_P.mm(torch.eye(H_P.size(1), dtype=torch.float32, device=device)))
        H_out_T = self.mlpT(H_T.mm(torch.eye(H_T.size(1), dtype=torch.float32, device=device)))

        H_mean_A = self.mlpA_mean(H_out_A)
        H_mean_C = self.mlpC_mean(H_out_C)
        H_mean_P = self.mlpP_mean(H_out_P)
        H_mean_T = self.mlpT_mean(H_out_T)
        H_std_A = self.mlpA_std(H_out_A)
        H_std_C = self.mlpC_std(H_out_C)
        H_std_P = self.mlpP_std(H_out_P)
        H_std_T = self.mlpT_std(H_out_T)
        return H_mean_A, H_mean_C, H_mean_P, H_mean_T, H_std_A, H_std_C, H_std_P, H_std_T


    def Reparametrization(self, H_mean_A, H_mean_C, H_mean_P, H_mean_T, H_std_A, H_std_C, H_std_P, H_std_T):
        eps_A = torch.randn_like(H_std_A, device=device)
        eps_C = torch.randn_like(H_std_C, device=device)
        eps_P = torch.randn_like(H_std_P, device=device)
        eps_T = torch.randn_like(H_std_T, device=device)

        std_A = torch.exp(H_std_A)
        std_C = torch.exp(H_std_C)
        std_P = torch.exp(H_std_P)
        std_T = torch.exp(H_std_T)

        LatentPresentation_A = eps_A.mul(std_A) + H_mean_A
        LatentPresentation_C = eps_C.mul(std_C) + H_mean_C
        LatentPresentation_P = eps_P.mul(std_P) + H_mean_P
        LatentPresentation_T = eps_T.mul(std_T) + H_mean_T

        LatentPresentation_A = self.transA(LatentPresentation_A)
        LatentPresentation_C = self.transA(LatentPresentation_C)
        LatentPresentation_P = self.transA(LatentPresentation_P)
        LatentPresentation_T = self.transA(LatentPresentation_T)
        return LatentPresentation_A, LatentPresentation_C, LatentPresentation_P, LatentPresentation_T

    def Pooling(self, H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T):
        shape_A = H_A.shape[1]
        shape_C = H_C.shape[1]
        shape_T = H_T.shape[1]
        listA = [1] * shape_A
        listC = [1] * shape_C
        listT = [1] * shape_T
        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)
        sum_list_C = []
        for tensor in torch.split(H_C, listC, dim=1):
            sum_pooling = torch.sum((H_mean_C * tensor), dim=0)
            sum_list_C.append(sum_pooling)
        sum_C = torch.stack(sum_list_C, dim=0)
        sum_list_T = []
        for tensor in torch.split(H_T, listT, dim=1):
            sum_pooling = torch.sum((H_mean_T * tensor), dim=0)
            sum_list_T.append(sum_pooling)
        sum_T = torch.stack(sum_list_T, dim=0)
        EventA_embedding = torch.cat((H_mean_P, sum_A), dim=1)
        EventC_embedding = torch.cat((H_mean_P, sum_C), dim=1)
        EventT_embedding = torch.cat((H_mean_P, sum_T), dim=1)
        return EventA_embedding, EventC_embedding, EventT_embedding

    def forward(self, H_A, H_C, H_P, H_T):
        H_mean_A, H_mean_C, H_mean_P, H_mean_T, H_std_A, H_std_C, H_std_P, H_std_T = self.Encoder(H_A, H_C, H_P, H_T)
        LatentPresentation_A, LatentPresentation_C, LatentPresentation_P, LatentPresentation_T = self.Reparametrization(H_mean_A, H_mean_C, H_mean_P, H_mean_T, H_std_A, H_std_C, H_std_P, H_std_T)
        EventA_embedding, EventC_embedding, EventT_embedding = self.Pooling(H_A, H_C, H_T, LatentPresentation_A, LatentPresentation_C, LatentPresentation_P, LatentPresentation_T)
        reconstructA = self.mlpEventsA(EventA_embedding)
        reconstructC = self.mlpEventsC(EventC_embedding)
        reconstructT = self.mlpEventsT(EventT_embedding)

        return LatentPresentation_A, LatentPresentation_C, LatentPresentation_P, LatentPresentation_T, reconstructA, reconstructC, reconstructT, H_mean_A, H_mean_C, H_mean_P, H_mean_T, H_std_A, H_std_C, H_std_P, H_std_T

class myModel_DBLP_noflow(nn.ModuleList):
    def __init__(self, d_inA, d_inC, d_inP, d_inT, d_inEvents, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out, flow_layers):
        '''

        :param d_inA: 作者个数
        :param d_inC: 会议个数
        :param d_inP: 论文个数
        :param d_inT: 关键词个数
        :param d_inEvents: 超边个数
        :param hidden_dim1: ecnoder维度
        :param hidden_dim2: 流模型维度
        :param flow_layers: ；流模型层数
        :param hidden_dim3: events重构隐藏层
        :param d_out: 最终输出维度
        '''
        super(myModel_DBLP_noflow, self).__init__()

        # 初始encoder 使用mlp 激活函数使用tanh
        self.mlpA = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpC = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpP = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpT = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        #投影
        self.transA = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transC = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transP = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        self.transT = torch.nn.Sequential(

            torch.nn.Tanh()
        )
        # self.mlpA_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpC_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpP_mean = torch.nn.Linear(hidden_dim1, hidden_dim)
        # self.mlpT_mean = torch.nn.Linear(hidden_dim1, hidden_dim)

        # 定义流模型
        # self.flowA = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # self.flowC = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # self.flowP = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # self.flowT = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # events重构 mlp,一共k - 1个
        self.mlpEventsA = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsC = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )
        self.mlpEventsT = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim3, d_inP)
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def Encoder(self, H_A, H_C, H_P, H_T):
        H_mean_A = self.mlpA(H_A.mm(torch.eye(H_A.size(1), dtype=torch.float32)))
        H_mean_C = self.mlpC(H_C.mm(torch.eye(H_C.size(1), dtype=torch.float32)))
        H_mean_P = self.mlpP(H_P.mm(torch.eye(H_P.size(1), dtype=torch.float32)))
        H_mean_T = self.mlpT(H_T.mm(torch.eye(H_T.size(1), dtype=torch.float32)))
        H_mean_A = self.transA(H_mean_A)
        H_mean_C = self.transC(H_mean_C)
        H_mean_P = self.transA(H_mean_P)
        H_mean_T = self.transC(H_mean_T)
        return H_mean_A, H_mean_C, H_mean_P, H_mean_T
    def PoolingSum(self, H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T):
        shape_A = H_A.shape[1]
        shape_C = H_C.shape[1]
        shape_T = H_T.shape[1]
        listA = [1] * shape_A
        listC = [1] * shape_C
        listT = [1] * shape_T
        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)
        sum_list_C = []
        for tensor in torch.split(H_C, listC, dim=1):
            sum_pooling = torch.sum((H_mean_C * tensor), dim=0)
            sum_list_C.append(sum_pooling)
        sum_C = torch.stack(sum_list_C, dim=0)
        sum_list_T = []
        for tensor in torch.split(H_T, listT, dim=1):
            sum_pooling = torch.sum((H_mean_T * tensor), dim=0)
            sum_list_T.append(sum_pooling)
        sum_T = torch.stack(sum_list_T, dim=0)
        EventA_embedding = torch.cat((H_mean_P, sum_A), dim=1)
        EventC_embedding = torch.cat((H_mean_P, sum_C), dim=1)
        EventT_embedding = torch.cat((H_mean_P, sum_T), dim=1)
        return EventA_embedding, EventC_embedding, EventT_embedding
    def PoolingMean(self, H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T):
        shape_A = H_A.shape[1]
        shape_C = H_C.shape[1]
        shape_T = H_T.shape[1]
        listA = [1.] * shape_A
        listC = [1.] * shape_C
        listT = [1.] * shape_T
        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            bias = torch.sum(tensor)
            if not torch.equal(bias, torch.Tensor(0, device=device)):
                mean_pooling = torch.div(sum_pooling, torch.sum(tensor))
                sum_list_A.append(mean_pooling)
            else:
                sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)

        sum_list_C = []
        for tensor in torch.split(H_C, listC, dim=1):
            sum_pooling = torch.sum((H_mean_C * tensor), dim=0)
            bias = torch.sum(tensor)
            if not torch.equal(bias, torch.Tensor(0, device=device)):
                mean_pooling = torch.div(sum_pooling, torch.sum(tensor))
                sum_list_C.append(mean_pooling)
            else:
                sum_list_C.append(sum_pooling)
        sum_C = torch.stack(sum_list_C, dim=0)
        sum_list_T = []

        for tensor in torch.split(H_T, listT, dim=1):
            sum_pooling = torch.sum((H_mean_T * tensor), dim=0)
            bias = torch.sum(tensor)
            if not torch.equal(bias, torch.Tensor(0, device=device)):
                mean_pooling = torch.div(sum_pooling, torch.sum(tensor))
                sum_list_T.append(mean_pooling)
            else:
                sum_list_C.append(sum_pooling)
        sum_T = torch.stack(sum_list_T, dim=0)
        EventA_embedding = torch.cat((H_mean_P, sum_A), dim=1)
        EventC_embedding = torch.cat((H_mean_P, sum_C), dim=1)
        EventT_embedding = torch.cat((H_mean_P, sum_T), dim=1)
        return EventA_embedding, EventC_embedding, EventT_embedding
    def forward(self, H_A, H_C, H_P, H_T):
        H_mean_A, H_mean_C, H_mean_P, H_mean_T = self.Encoder(H_A, H_C, H_P, H_T)
        # u_A, log_det_jacobianA = self.flowA(H_mean_A)
        # u_C, log_det_jacobianC = self.flowC(H_mean_C)
        # u_P, log_det_jacobianP = self.flowP(H_mean_P)
        # u_T, log_det_jacobianT = self.flowT(H_mean_T)
        # predicted_A = self.flowA.inverse(u_A)
        # predicted_C = self.flowC.inverse(u_C)
        # predicted_P = self.flowA.inverse(u_P)
        # predicted_T = self.flowT.inverse(u_T)
        EventA_embedding, EventC_embedding, EventT_embedding = self.PoolingSum(H_A, H_C, H_T, H_mean_A, H_mean_C, H_mean_P, H_mean_T)
        # reconstructA = self.mlpEventsA(EventA_embedding)
        # reconstructC = self.mlpEventsC(EventC_embedding)
        # reconstructT = self.mlpEventsT(EventT_embedding)
        EventA_embedding = self.proj(EventA_embedding)
        EventT_embedding = self.proj(EventT_embedding)
        EventC_embedding = self.proj(EventC_embedding)
        return EventA_embedding,EventC_embedding, EventT_embedding, H_mean_A, H_mean_C, H_mean_P, H_mean_T


class myModel_ICDM(nn.Module):
    def __init__(self, d_inA, d_inD, d_inM,  d_inEvents, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, d_out, flow_layers):
        '''

        :param d_inA: 作者个数
        :param d_inC: 会议个数
        :param d_inP: 论文个数
        :param d_inT: 关键词个数
        :param d_inEvents: 超边个数
        :param hidden_dim1: ecnoder维度
        :param hidden_dim2: 流模型维度
        :param flow_layers: ；流模型层数
        :param hidden_dim3: events重构隐藏层
        :param d_out: 最终输出维度
        '''
        super(myModel_ICDM, self).__init__()

        # 初始encoder 使用mlp 激活函数使用tanh
        self.mlpA = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpD = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )

        self.mlpM = torch.nn.Sequential(
            torch.nn.Linear(d_inEvents, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim)
        )



        # 定义流模型
        # self.flowA = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # self.flowD = RealNVP(flow_layers, hidden_dim, hidden_dim2)
        # self.flowM = RealNVP(flow_layers, hidden_dim, hidden_dim2)

        # events重构 mlp,一共k - 1个
        self.mlpEventsA = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inM)
        )
        self.mlpEventsD = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim3, d_inM)
        )

    def Encoder(self, H_A, H_D, H_M):
        H_mean_A = self.mlpA(H_A.mm(torch.eye(H_A.size(1), dtype=torch.float32)))
        H_mean_D = self.mlpD(H_D.mm(torch.eye(H_D.size(1), dtype=torch.float32)))
        H_mean_M = self.mlpM(H_M.mm(torch.eye(H_M.size(1), dtype=torch.float32)))

        return H_mean_A, H_mean_D, H_mean_M
    def Pooling(self, H_A, H_D, H_M, H_mean_A, H_mean_D, H_mean_M):
        shape_A = H_A.shape[1]
        shape_D = H_D.shape[1]

        listA = [1] * shape_A
        listD = [1] * shape_D

        sum_list_A = []
        for tensor in torch.split(H_A, listA, dim=1):
            sum_pooling = torch.sum((H_mean_A * tensor), dim=0)
            sum_list_A.append(sum_pooling)
        sum_A = torch.stack(sum_list_A, dim=0)
        sum_list_D = []
        for tensor in torch.split(H_D, listD, dim=1):
            sum_pooling = torch.sum((H_mean_D * tensor), dim=0)
            sum_list_D.append(sum_pooling)
        sum_D = torch.stack(sum_list_D, dim=0)

        EventA_embedding = torch.cat((H_mean_M, sum_A), dim=1)
        EventD_embedding = torch.cat((H_mean_M, sum_D), dim=1)

        return EventA_embedding, EventD_embedding

    def forward(self, H_A, H_D, H_M):
        H_mean_A, H_mean_D, H_mean_M = self.Encoder(H_A, H_D, H_M)
        # u_A, log_det_jacobianA = self.flowA(H_mean_A)
        # u_D, log_det_jacobianD = self.flowD(H_mean_D)
        # u_M, log_det_jacobianM = self.flowM(H_mean_M)

        # predicted_A = self.flowA.inverse(u_A)
        # predicted_D = self.flowD.inverse(u_D)
        # predicted_M = self.flowM.inverse(u_M)

        EventA_embedding, EventD_embedding = self.Pooling(H_A, H_D, H_M, H_mean_A, H_mean_D, H_mean_M)
        reconstructA = self.mlpEventsA(EventA_embedding)
        reconstructD = self.mlpEventsD(EventD_embedding)

        return reconstructA, reconstructD, H_mean_A, H_mean_D, H_mean_M



