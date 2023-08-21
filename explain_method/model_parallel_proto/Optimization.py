import torch
import torch.nn as nn
import numpy as np
import time
from scipy.spatial.distance import cdist
from utils import plot_figure_effects_SA
from utils import euclidian_dist

class Optimization_experimetns:

    def __init__(self, epochs, dim, lr, device, nbr_num):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.early_stop = True
        self.alpha = 1
        self.beta = 0
        self.gamma = 0
        self.dim = dim
        self.nbr_num = nbr_num

    def explain(self, explain_instance, explain_label, proto_dict):

        print('attack:',explain_label)

        begin_time = time.time()

        centers = np.array([proto.center for proto in proto_dict[0]])
        sizes = np.array([proto.size for proto in proto_dict[0]])
        sizes = np.log(sizes) / np.sum(np.log(sizes))


        w_init = torch.from_numpy(np.ones(len(explain_instance)) / np.sqrt(self.dim)).type(torch.float).to(self.device)
        w = w_init.clone().detach()
        explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
        centers = torch.from_numpy(centers).type(torch.float).to(self.device)
        sizes = torch.from_numpy(sizes).type(torch.float).to(self.device)

        with torch.no_grad():

            dist = torch.sqrt(torch.sum((explain_instance - centers) ** 2, axis=1))
            min_dist_idx = torch.argsort(dist)[:self.nbr_num].cpu().data.numpy()

            nearest_proto = centers[min_dist_idx]
            nearest_proto_size = sizes[min_dist_idx]
            remain_idx = np.array(list(set(np.arange(len(centers))) - set(min_dist_idx)))

            remain_proto = centers[remain_idx]
            remain_proto_size = sizes[remain_idx]

        w.requires_grad = True
        optimizer = torch.optim.SGD([w], lr=self.lr)

        for epoch in range(self.epochs):

            # if epoch % 25 == 0:
            #     random_idx = np.random.choice(remain_idx, np.max([1, int(len(remain_idx)/3)]), replace=False)
            #     remain_proto = centers[random_idx]
            #     remain_proto_size = sizes[random_idx]

            a = torch.sqrt(torch.sum((w * explain_instance - w * remain_proto) ** 2, dim=1))
            b = torch.sqrt(torch.sum((w * explain_instance - w * nearest_proto) ** 2, dim=1))

            Loss1 = torch.mean(remain_proto_size * torch.exp(-a))
            Loss2 = torch.mean(nearest_proto_size * torch.exp(-b))
            Loss3 = (torch.norm(w, p=2) + torch.norm(w, p=1))/len(w)

            Loss = self.alpha * Loss1 + self.beta * Loss2 + self.gamma * Loss3

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            if epoch % 100 == 0 and epoch!=0:
                print('{}:'.format(epoch+1),
                      'Final loss: %.7f,' % Loss.item(),
                      'Loss1: %.7f,' % Loss1.item(),
                      'Loss2: %.7f,' % Loss2.item(),
                      'Loss3: %.7f,' % Loss3.item()
                      )

        end_time = time.time()

        w_numpy = w.cpu().data.numpy()
        print(w_numpy)
        print('------------------------------------------')
        # print('Finish Interpretation after {} steps'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
        #       "Time elasped: %.2fs)" % (end_time - begin_time))



        return w_numpy








# class Optimization_experimetns:
#
#     def __init__(self, epochs, dim, lr, device, nbr_num):
#         self.epochs = epochs
#         self.lr = lr
#         self.device = device
#         self.early_stop = True
#         self.alpha = 1
#         self.eps = 1e-4
#         self.dim = dim
#         self.nbr_num = nbr_num
#
#     def explain(self, explain_instance, explain_label, proto_dict):
#         begin_time = time.time()
#
#         center_lst = []
#         size_lst = []
#         ratio_lst = []
#         for label_key in proto_dict:
#             if label_key != explain_label:
#                 center_array = np.array([proto.center for proto in proto_dict[label_key]])
#                 size_array = np.array([proto.size for proto in proto_dict[label_key]])
#                 ratio_lst.append(np.sum(size_array))
#                 size_array = (1 / (np.log(size_array) + self.eps)) / (np.sum(1 / (np.log(size_array)+self.eps)))
#                 center_lst.append(center_array)
#                 size_lst.append(size_array)
#
#         # ratio_array = (1 / np.array(ratio_lst)) / (np.sum(1 / np.array(ratio_lst)))
#         ratio_array = np.array(ratio_lst) / np.sum(np.array(ratio_lst))
#
#         w_init = torch.from_numpy(np.ones(len(explain_instance))).type(torch.float).to(self.device)
#         w = w_init.clone().detach()
#         explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
#         center_tensor_lst = [torch.from_numpy(center_array).type(torch.float).to(self.device) for center_array in center_lst]
#         size_tensor_lst = [torch.from_numpy(size_array).type(torch.float).to(self.device) for size_array in size_lst]
#
#
#         with torch.no_grad():
#
#             nearest_proto_lst = []
#             nearest_proto_size_lst = []
#             random_proto_lst = []
#             random_proto_size_lst = []
#
#             for i in range(len(center_tensor_lst)):
#                 center_tensor = center_tensor_lst[i]
#                 size_tensor = size_tensor_lst[i]
#                 dist = torch.sum((explain_instance - center_tensor) ** 2, axis=1)
#                 min_dist_idx = torch.argsort(dist)[:self.nbr_num]
#                 nearest_proto = center_tensor[min_dist_idx]
#                 nearest_proto_size = size_tensor[min_dist_idx]
#
#                 remain_idx = np.array(list(set(np.arange(len(center_tensor))) - set(min_dist_idx)))
#                 remain_random_idx = np.random.choice(remain_idx, int(len(remain_idx)/2), replace=False)
#                 random_proto = center_tensor[remain_random_idx]
#                 random_proto_size = size_tensor[remain_random_idx]
#
#                 nearest_proto_lst.append(nearest_proto)
#                 nearest_proto_size_lst.append(nearest_proto_size)
#                 random_proto_lst.append(random_proto)
#                 random_proto_size_lst.append(random_proto_size)
#
#
#         relu = nn.ReLU()
#         w.requires_grad = True
#         optimizer = torch.optim.SGD([w], lr=self.lr)
#
#         for epoch in range(self.epochs):
#
#             Loss1 = 0
#             Loss2 = 0
#             Loss3 = 0
#
#             for i in range(len(nearest_proto_lst)):
#
#                 a = torch.sqrt(torch.sum((w * (explain_instance - nearest_proto_lst[i])) ** 2, dim=1))
#                 b = torch.sqrt(torch.sum((w * (explain_instance - random_proto_lst[i])) ** 2, dim=1))
#
#                 Loss1 += ratio_array[i] * torch.mean(torch.exp(-a))
#                 Loss2 += ratio_array[i] * torch.mean(torch.exp(-b))
#                 Loss3 += ratio_array[i] * (torch.exp(relu(torch.mean(a) - torch.mean(b) + self.eps)) - 1)
#
#             Loss = Loss1 + Loss2 + Loss3
#
#             optimizer.zero_grad()
#             Loss.backward()
#             optimizer.step()
#
#             if epoch % 50 == 0:
#                 print(f'epoch={epoch + 1}/{self.epochs}, Loss1={Loss1}, Loss2={Loss2}, Loss3={Loss3}')
#
#             if epoch != 0 and epoch % 10 == 0:
#                 with torch.no_grad():
#                     nearest_proto_lst = []
#                     nearest_proto_size_lst = []
#                     random_proto_lst = []
#                     random_proto_size_lst = []
#
#                     for i in range(len(center_tensor_lst)):
#                         center_tensor = center_tensor_lst[i]
#                         size_tensor = size_tensor_lst[i]
#                         dist = torch.sum((explain_instance - w * center_tensor) ** 2, axis=1)
#                         min_dist_idx = torch.argsort(dist)[:self.nbr_num]
#                         nearest_proto = center_tensor[min_dist_idx]
#                         nearest_proto_size = size_tensor[min_dist_idx]
#
#                         remain_idx = np.array(list(set(np.arange(len(center_tensor))) - set(min_dist_idx)))
#                         remain_random_idx = np.random.choice(remain_idx, int(len(remain_idx)/2), replace=False)
#                         random_proto = center_tensor[remain_random_idx]
#                         random_proto_size = size_tensor[remain_random_idx]
#
#                         nearest_proto_lst.append(nearest_proto)
#                         nearest_proto_size_lst.append(nearest_proto_size)
#                         random_proto_lst.append(random_proto)
#                         random_proto_size_lst.append(random_proto_size)
#
#
#
#         end_time = time.time()
#
#         w_numpy = w.cpu().data.numpy()
#
#
#
#         # print('Finish Interpretation after {} steps'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
#         #       "Time elasped: %.2fs)" % (end_time - begin_time))
#
#
#
#         return w_numpy


















# class Optimization:
#
#     def __init__(self, epochs, dim, lr, device, nbr_num):
#         self.epochs = epochs
#         self.lr = lr
#         self.device = device
#         self.early_stop = True
#         self.alpha = 1
#         self.eps = 1e-4
#         self.dim = dim
#         self.nbr_num = nbr_num
#
#     def explain(self, explain_instance, explain_label, proto_dict):
#         begin_time = time.time()
#
#         center_lst = []
#         size_lst = []
#         for label_key in proto_dict:
#             if label_key != explain_label:
#                 center_array = np.array([proto.center for proto in proto_dict[label_key]])
#                 size_array = np.array([proto.size for proto in proto_dict[label_key]])
#                 size_array = (1/size_array) / (np.sum(1/size_array))
#
#                 center_lst.append(center_array)
#                 size_lst.append(size_array)
#
#         w_init = torch.from_numpy(np.ones(len(explain_instance))).type(torch.float).to(self.device)
#         w = w_init.clone().detach()
#         explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
#         center_tensor_lst = [torch.from_numpy(center_array).type(torch.float).to(self.device) for center_array in center_lst]
#         size_tensor_lst = [torch.from_numpy(size_array).type(torch.float).to(self.device) for size_array in size_lst]
#
#
#         with torch.no_grad():
#
#             nearest_proto_lst = []
#             nearest_proto_size_lst = []
#             random_proto_lst = []
#             random_proto_size_lst = []
#
#             for i in range(len(center_tensor_lst)):
#                 center_tensor = center_tensor_lst[i]
#                 size_tensor = size_tensor_lst[i]
#                 dist = torch.sum((explain_instance - center_tensor) ** 2, axis=1)
#                 min_dist_idx = torch.argsort(dist)[:self.nbr_num]
#                 nearest_proto = center_tensor[min_dist_idx]
#                 nearest_proto_size = size_tensor[min_dist_idx]
#
#                 remain_idx = np.array(list(set(np.arange(len(center_tensor))) - set(min_dist_idx)))
#                 remain_random_idx = np.random.choice(remain_idx, int(len(remain_idx)/2), replace=False)
#                 random_proto = center_tensor[remain_random_idx]
#                 random_proto_size = size_tensor[remain_random_idx]
#
#                 nearest_proto_lst.append(nearest_proto)
#                 nearest_proto_size_lst.append(nearest_proto_size)
#                 random_proto_lst.append(random_proto)
#                 random_proto_size_lst.append(random_proto_size)
#
#
#         relu = nn.ReLU()
#         w.requires_grad = True
#         optimizer = torch.optim.SGD([w], lr=self.lr)
#
#         for epoch in range(self.epochs):
#
#             Loss1 = 0
#             Loss2 = 0
#             Loss3 = 0
#
#             for i in range(len(nearest_proto_lst)):
#
#                 a = torch.sqrt(torch.sum((w * (explain_instance - nearest_proto_lst[i])) ** 2, dim=1))
#                 b = torch.sqrt(torch.sum((w * (explain_instance - random_proto_lst[i])) ** 2, dim=1))
#
#                 Loss1 += torch.mean(torch.exp(- nearest_proto_size_lst[i] * a))
#                 Loss2 += torch.mean(torch.exp(- random_proto_size_lst[i] * b))
#                 Loss3 += torch.exp(relu(torch.mean(a) - torch.mean(b) + self.eps)) - 1
#
#             Loss = Loss1 + Loss2 + Loss3
#
#             optimizer.zero_grad()
#             Loss.backward()
#             optimizer.step()
#
#             if epoch % 50 == 0:
#                 print(f'epoch={epoch + 1}/{self.epochs}, Loss1={Loss1}, Loss2={Loss2}, Loss3={Loss3}')
#
#             if epoch != 0 and epoch % 10 == 0:
#                 with torch.no_grad():
#                     nearest_proto_lst = []
#                     nearest_proto_size_lst = []
#                     random_proto_lst = []
#                     random_proto_size_lst = []
#
#                     for i in range(len(center_tensor_lst)):
#                         center_tensor = center_tensor_lst[i]
#                         size_tensor = size_tensor_lst[i]
#                         dist = torch.sum((explain_instance - w * center_tensor) ** 2, axis=1)
#                         min_dist_idx = torch.argsort(dist)[:self.nbr_num]
#                         nearest_proto = center_tensor[min_dist_idx]
#                         nearest_proto_size = size_tensor[min_dist_idx]
#
#                         remain_idx = np.array(list(set(np.arange(len(center_tensor))) - set(min_dist_idx)))
#                         remain_random_idx = np.random.choice(remain_idx, int(len(remain_idx)/2), replace=False)
#                         random_proto = center_tensor[remain_random_idx]
#                         random_proto_size = size_tensor[remain_random_idx]
#
#                         nearest_proto_lst.append(nearest_proto)
#                         nearest_proto_size_lst.append(nearest_proto_size)
#                         random_proto_lst.append(random_proto)
#                         random_proto_size_lst.append(random_proto_size)
#
#
#
#         end_time = time.time()
#
#         w_numpy = w.cpu().data.numpy()
#
#
#
#         # print('Finish Interpretation after {} steps'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
#         #       "Time elasped: %.2fs)" % (end_time - begin_time))
#
#
#
#         return w_numpy


























