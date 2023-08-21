import torch
import numpy as np
import time
import torch.nn as nn

import torch.nn.functional as F
from utils import MatrixEuclideanDistances,str_to_bool
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)



def MatrixEuclideanDistances_tensor(a, b):

    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()

    return torch.clamp(sum_sq_a+sum_sq_b-2*a.mm(bt), min=0.)

class OPT_Explain_fu:

    def __init__(self, epochs, lr, alpha,  eps, t1, t2, dim, device):

        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.t1 = t1
        self.t2 = t2
        self.device = device
        self.early_stop = True
        self.dim = dim
        self.reparam_f = False
        self.reparam_p = False

    def concrete_transformation(self, p):
        if self.reparam_f:
            reverse_theta = torch.ones_like(p) - p
            unif_noise = torch.Tensor(len(p)).uniform_(0, 1).type(torch.float).to(self.device)
            reverse_unif_noise = torch.ones_like(unif_noise) - unif_noise
            appro = torch.log(p + self.eps) - torch.log(reverse_theta + self.eps) + torch.log(unif_noise) - torch.log(
                reverse_unif_noise)
        else:
            reverse_theta = torch.ones_like(p) - p
            appro = torch.log(p + self.eps) - torch.log(reverse_theta + self.eps)

        logit = appro / self.t1

        return torch.sigmoid(logit)


        data_metric_weight_new = feature_importance * data_metric_array + (1 - feature_importance) * explain_instance
        dist_matrix_weight = MatrixEuclideanDistances(data_metric_weight_new, centers)
        in_circle_weight_array = np.zeros(len(dist_matrix_weight))
        for j in range(len(dist_matrix_weight)):
            cover_idx = np.where(dist_matrix_weight[j] <= rs ** 2 + self.eps)[0]
            if len(cover_idx) == 0:
                in_circle_weight = 0
            else:
                in_circle_weight = 1
            in_circle_weight_array[j] = in_circle_weight



        return [round(x,4) for x in feat_lst_results],  [round(x,4) for x in imp_fea_num_lst], np.mean(in_circle_weight_array)



    # training on centers
    @staticmethod
    def inverse_gumbel_cdf(y, mu, beta):
        return mu - beta * np.log(-np.log(y))

    def gumbel_softmax_sampling(self, h, mu=0, beta=1, tau=0.1):
        """
        h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
        """
        shape_h = h.size()
        p = F.softmax(h, dim=1)
        y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
        g = self.inverse_gumbel_cdf(y, mu, beta).to(self.device)
        x = torch.log(p) + g  # samples follow Gumbel distribution.
        # using softmax to generate one_hot vector:
        x = x / tau
        x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
        return x

    def opt_exp(self,  explain_instance, explain_label, proto_array, w_fea):

        center_array = np.array([proto.center for proto in proto_array])
        r_array = np.array([proto.r for proto in proto_array])
        size_array = np.array([proto.size for proto in proto_array])
        size_ratio_array = size_array / np.sum(size_array)

        explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
        center_tensor = torch.from_numpy(center_array).type(torch.float).to(self.device)
        r_tensor = torch.from_numpy(r_array).type(torch.float).to(self.device)
        size_ratio_tensor = torch.from_numpy(size_ratio_array).type(torch.float).to(self.device)

        w_proto = torch.ones(size=(len(center_array), len(center_array))).type(torch.float).to(
            self.device)

        # 优化圆的mask
        relu = nn.ReLU()
        w_fea.requires_grad = True

        optimizer = torch.optim.Adam([w_fea], lr=self.lr)

        begin_time = time.time()
        Last_loss = np.Inf
        tau = 1
        for epoch in range(self.epochs):
            w_fea_normalize = torch.sigmoid(w_fea)
            mask_fea = self.concrete_transformation(w_fea_normalize)
            mask_reverse = torch.ones_like(mask_fea) - mask_fea
            x_new = mask_reverse * explain_instance + mask_fea * center_tensor
            mask_proto = w_proto

            dist_matrix = MatrixEuclideanDistances_tensor(x_new, center_tensor)
            instance_dist_tensor = torch.sqrt(dist_matrix + self.eps) - r_tensor.view(1,-1) + self.eps
            instance_dist_tensor_relu = relu(instance_dist_tensor)
            dist = torch.sum(torch.mul(mask_proto, instance_dist_tensor_relu), dim=1)
            Loss1 = torch.sum(size_ratio_tensor * dist)
            Loss2 = torch.norm(mask_fea, p=1) / self.dim
            Loss = Loss1 + self.alpha * Loss2
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()


            if torch.abs(Last_loss - Loss) < 1e-4:
                break
            Last_loss = Loss


        end_time = time.time()
        w_numpy = mask_fea.cpu().data.numpy()
        w_fea_numpy = w_fea.cpu().data.numpy()
        instance_dist_numpy = instance_dist_tensor_relu.cpu().data.numpy()


        return w_numpy, w_fea_numpy, epoch, instance_dist_numpy




