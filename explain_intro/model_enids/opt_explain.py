import torch
import numpy as np
import time
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as Data  # jin xing xiao pi xun lian de mo kuai
import torch.optim as optim
from scipy import spatial
import torch.nn.functional as F
from utils import MatrixEuclideanDistances,str_to_bool
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)



class MYSELFDATASET(Dataset):


    def __init__(self, centers, r, size):
        super(Dataset, self).__init__()

        self.centers = centers
        self.r = r
        self.size = size


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """

        centers_sample = self.centers[index]
        r_sample = self.r[index]
        size_sample = self.size[index]


        return centers_sample,r_sample, size_sample, index

    def __len__(self):
        return len(self.centers)


def MatrixEuclideanDistances_tensor(a, b):

    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()

    return torch.clamp(sum_sq_a+sum_sq_b-2*a.mm(bt), min=0.)

class OPT_Explain:

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


    def concrete_transformation(self, p):
        # if self.reparam_f:
        #     reverse_theta = torch.ones_like(p) - p
        #     unif_noise = torch.Tensor(len(p)).uniform_(0, 1).type(torch.float).to(self.device)
        #     reverse_unif_noise = torch.ones_like(unif_noise) - unif_noise
        #     appro = torch.log(p + self.eps) - torch.log(reverse_theta + self.eps) + torch.log(unif_noise) - torch.log(
        #         reverse_unif_noise)
        # else:
        reverse_theta = torch.ones_like(p) - p
        appro = torch.log(p + self.eps) - torch.log(reverse_theta + self.eps)

        logit = appro / self.t1

        return torch.sigmoid(logit)


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

    def opt_exp(self,  explain_instance, explain_label, proto_array, w_fea, w_proto):

        # print('attack:', explain_label)
        center_array = np.array([proto.center for proto in proto_array])
        r_array = np.array([proto.r for proto in proto_array])
        size_array = np.array([proto.size for proto in proto_array])
        size_ratio_array = size_array / np.sum(size_array)

        explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
        center_tensor = torch.from_numpy(center_array).type(torch.float).to(self.device)
        r_tensor = torch.from_numpy(r_array).type(torch.float).to(self.device)
        size_ratio_tensor = torch.from_numpy(size_ratio_array).type(torch.float).to(self.device)


        # 优化圆的mask
        relu = nn.ReLU()
        # w_fea = torch.rand(len(explain_instance)).type(torch.float).to(self.device)
        # w_fea = torch.tensor(w_fea_init).type(torch.float).to(self.device)
        # w_proto = torch.rand(size=(len(center_tensor), len(center_tensor))).type(torch.float).to(self.device)

        w_fea.requires_grad = True
        w_proto.requires_grad = True

        optimizer = torch.optim.Adam([w_fea, w_proto], lr=self.lr)

        begin_time = time.time()
        Last_loss = np.Inf
        tau = 1
        for epoch in range(self.epochs):
            w_fea_normalize = torch.sigmoid(w_fea)
            mask_fea = self.concrete_transformation(w_fea_normalize)
            mask_reverse = torch.ones_like(mask_fea) - mask_fea
            x_new = mask_reverse * explain_instance + mask_fea * center_tensor
            w_proto_normalize = torch.sigmoid(w_proto)
            # mask_proto = F.softmax(w_proto_normalize / self.t2, dim=1)
            # if self.reparam_p:
            #     if epoch % 100 == 0:
            #         tau = tau * 0.5
            #     mask_proto = self.gumbel_softmax_sampling(w_proto_normalize, tau=tau)
            # else:
            mask_proto = F.softmax(w_proto_normalize / self.t2, dim=1)


            dist_matrix = MatrixEuclideanDistances_tensor(x_new, center_tensor)
            # dist_matrix = torch.ones((len(x_new), len(center_tensor))).to(self.device)
            # compensation = torch.sum((mask_fea.view(1,-1)) ** 2 * (ss_tensor / size_tensor.view(-1,1) - center_tensor ** 2), dim=1).view(-1,1)
            instance_dist_tensor = torch.sqrt(dist_matrix + self.eps) - r_tensor.view(1,-1) + self.eps
            instance_dist_tensor_relu = relu(instance_dist_tensor)
            dist = torch.sum(torch.mul(mask_proto, instance_dist_tensor_relu), dim=1)
            Loss1 = torch.sum(size_ratio_tensor * dist)
            Loss2 = torch.norm(mask_fea, p=1) / self.dim
            Loss = Loss1 + self.alpha * Loss2
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            # with torch.no_grad():
                # if epoch % 100 == 0 or epoch == 10 or epoch == self.epochs - 1:
                #
                #    print(
                #         f'| Epoch: {epoch + 1:02}/{self.epochs:02}|'
                #         f'Loss: {Loss:.4f}|'
                #         f'Loss1: {Loss1:.4f}|'
                #         f'Loss2: {Loss2:.4f}|'
                #    )
                # if epoch == self.epochs-1:
                #     explain_instance_npy = explain_instance.cpu().data.numpy()
                #     a, b, c = self.results_print(explain_instance_npy,
                #                                  mask_fea.cpu().data.numpy(),
                #                                  center_array,
                #                                  r_array,
                #                                  data_train)
                #     print(a, b, c)
                #
                #
                #     for j in range(10):
                #        idx = torch.argsort(-mask_proto[j])[:10]
                #        mask_proto_array = mask_proto[j, idx].cpu().data.numpy()
                #        ins_dist_array = relu(instance_dist_tensor[j, idx]).cpu().data.numpy()
                #        print(j, size_ratio_array[j],
                #               round(np.sum(mask_proto_array * ins_dist_array), 2),
                #               'mask_proto=', [round(item, 3) for item in mask_proto_array],
                #               'ins_dist_tensor=', [round(item, 3) for item in ins_dist_array])
                #        # dist_array = dist.cpu().data.numpy()
                #        # idx = np.where(dist_array!=0)[0]
                #        # print(dist_array[idx], len(idx))
                #     print('----------------------------------------------------------------')
            if torch.abs(Last_loss - Loss) < 1e-4:
                break
            Last_loss = Loss


        end_time = time.time()
        w_numpy = mask_fea.cpu().data.numpy()
        w_fea_numpy = w_fea.cpu().data.numpy()
        w_proto_numpy = w_proto.cpu().data.numpy()
        instance_dist_numpy = instance_dist_tensor_relu.cpu().data.numpy()
        # print(w_numpy, np.argsort(-w_numpy))

        # print('Finish Interpretation after {} epoch'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
        #       "Time elasped: %.2fs)" % (end_time - begin_time))

        return w_numpy, w_fea_numpy, w_proto_numpy, epoch, instance_dist_numpy

    # def opt_exp(self,  explain_instance, explain_label, proto_array, data_train):
    #
    #     print('attack:', explain_label)
    #     center_array = np.array([proto.center for proto in proto_array])
    #     r_array = np.array([proto.r for proto in proto_array])
    #     size_array = np.array([proto.size for proto in proto_array])
    #     size_ratio_array = size_array / np.sum(size_array)
    #
    #     ss_array = np.array([proto.ss for proto in proto_array])
    #
    #
    #     explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
    #     center_tensor = torch.from_numpy(center_array).type(torch.float).to(self.device)
    #     r_tensor = torch.from_numpy(r_array).type(torch.float).to(self.device)
    #     size_tensor = torch.from_numpy(size_array).type(torch.float).to(self.device)
    #     size_ratio_tensor = torch.from_numpy(size_ratio_array).type(torch.float).to(self.device)
    #     ss_tensor = torch.from_numpy(ss_array).type(torch.float).to(self.device)
    #     data_train_tensor = torch.from_numpy(data_train).type(torch.float).to(self.device)
    #
    #
    #     # 优化圆的mask
    #     relu = nn.ReLU()
    #
    #
    #     w_fea = torch.rand(len(explain_instance)).type(torch.float).to(self.device)
    #     w_proto = torch.rand(size=(len(data_train_tensor), len(center_tensor))).type(torch.float).to(self.device)
    #
    #     w_fea.requires_grad = True
    #     w_proto.requires_grad = True
    #
    #     optimizer = torch.optim.Adam([w_fea, w_proto], lr=self.lr)
    #
    #     begin_time = time.time()
    #     Last_loss = np.Inf
    #     for epoch in range(self.epochs):
    #         w_fea_normalize = torch.sigmoid(w_fea)
    #         mask_fea = self.concrete_transformation(w_fea_normalize)
    #         mask_reverse = torch.ones_like(mask_fea) - mask_fea
    #         x_new = mask_reverse * explain_instance + mask_fea * data_train_tensor
    #         w_proto_normalize = torch.sigmoid(w_proto)
    #         mask_proto = F.softmax(w_proto_normalize / self.t2, dim=1)
    #         dist_matrix = MatrixEuclideanDistances_tensor(x_new, center_tensor)
    #         compensation = 0
    #         # compensation = torch.sum((mask_fea.view(1,-1)) ** 2 * (ss_tensor / size_tensor.view(-1,1) - center_tensor ** 2), dim=1).view(-1,1)
    #         instance_dist_tensor = torch.sqrt(dist_matrix + compensation + self.eps) - r_tensor.view(1,-1) + self.eps
    #         dist = torch.sum(torch.mul(mask_proto, relu(instance_dist_tensor)), dim=1)
    #         Loss1 = torch.mean(dist)
    #         Loss2 = torch.norm(mask_fea, p=1) / self.dim
    #         # Loss3 = torch.norm(mask_proto, p='nuc')
    #         Loss = Loss1 + self.alpha * Loss2
    #         optimizer.zero_grad()
    #         Loss.backward()
    #         optimizer.step()
    #
    #         with torch.no_grad():
    #             if epoch % 100 == 0 or epoch == 10 or epoch == self.epochs - 1:
    #
    #                print(
    #                     f'| Epoch: {epoch + 1:02}/{self.epochs:02}|'
    #                     f'Loss: {Loss:.4f}|'
    #                     f'Loss1: {Loss1:.4f}|'
    #                     f'Loss2: {Loss2:.4f}|'
    #                     # f'Loss3: {Loss3:.4f}|',
    #                )
    #             # if epoch == self.epochs-1:
    #             #     explain_instance_npy = explain_instance.cpu().data.numpy()
    #             #     a, b, c = self.results_print(explain_instance_npy,
    #             #                                  mask_fea.cpu().data.numpy(),
    #             #                                  center_array,
    #             #                                  r_array,
    #             #                                  data_train)
    #             #     print(a, b, c)
    #             #
    #             #
    #             #     for j in range(10):
    #             #        idx = torch.argsort(-mask_proto[j])[:10]
    #             #        mask_proto_array = mask_proto[j, idx].cpu().data.numpy()
    #             #        ins_dist_array = relu(instance_dist_tensor[j, idx]).cpu().data.numpy()
    #             #        print(j, size_ratio_array[j],
    #             #               round(np.sum(mask_proto_array * ins_dist_array), 2),
    #             #               'mask_proto=', [round(item, 3) for item in mask_proto_array],
    #             #               'ins_dist_tensor=', [round(item, 3) for item in ins_dist_array])
    #             #        # dist_array = dist.cpu().data.numpy()
    #             #        # idx = np.where(dist_array!=0)[0]
    #             #        # print(dist_array[idx], len(idx))
    #             #     print('----------------------------------------------------------------')
    #
    #         if torch.abs(Loss - Last_loss) < 1e-4:
    #             break
    #
    #         Last_loss = Loss
    #
    #     end_time = time.time()
    #
    #
    #     w_numpy = mask_fea.cpu().data.numpy()
    #     # imp_idx_len = b
    #     print(w_numpy, np.argsort(-w_numpy))
    #     # print('imp_fea_num=', imp_idx_len)
    #     feature_importance = w_numpy
    #
    #     print('Finish Interpretation after {} epoch'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
    #           "Time elasped: %.2fs)" % (end_time - begin_time))
    #
    #     return feature_importance




