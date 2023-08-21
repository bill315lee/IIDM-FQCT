'''This code is test on the nearest prototypes, which finds the root cause'''

import argparse
import torch
import numpy as np
import time
import torch.nn as nn

from tqdm import tqdm
import os
from collections import Counter
import sys
sys.path.append('/home/lb/project/explain_ids')
from utils import gen_pseudo_points_in_proto
from base_active_classifier.Prototypes import Prototypes
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 15))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8)) # figure width high
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=80, c=palette[colors.astype(np.int)])
    plt.xlim(0, 1) #横纵坐标
    plt.ylim(0, 1)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(15):
        # Position of each label.
        if i in colors:
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    return f, ax, sc, txts

def tsne_plot(X,y,path="",size=8):
    print("Plotting")

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})


    digits_proj = TSNE(random_state=1, n_jobs=-1).fit_transform(X)
    scatter(digits_proj, y)
    plt.savefig(path, dpi=120)


def euclidian_dist(a, b):
    if a.ndim == 1:
        a = a.reshape(1,-1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def read_train_test_data(args):

    if args.datatype == 'IDS2017':
        prefix_path = os.path.join('/sda', 'bill', args.datatype, args.datatype + '_fix', 'remain')
        data_path = f'{prefix_path}/{args.train_date}_drop_data.npy'
        label_path = f'{prefix_path}/{args.train_date}_drop_label.npy'
        four_tag_path = f'{prefix_path}/{args.train_date}_drop_four_tag.npy'

    elif args.datatype == 'NADC':
        file_name = f'{args.train_date}_labelprocess'
        data_path = os.path.join('/sda', 'bill', 'NADC', file_name + '.npz')
        label_path = None
        four_tag_path = None

    elif args.datatype == 'toy' and args.train_date in ['1', '2', '3']:
        file_name = f'{args.datatype}{args.train_date}'
        data_path = os.path.join('/home', 'lb', 'project', 'explain_ids', 'toydata', file_name + '.npz')
    else:
        raise NotImplementedError


    if args.datatype == "IDS2017":

        data_end = np.load(data_path)
        labels = np.load(label_path, allow_pickle=True)
        four_tag = np.load(four_tag_path, allow_pickle=True)

        data_mu = np.mean(data_end, axis=0)
        data_std = np.std(data_end, axis=0)
        data_process = (data_end - data_mu) / (data_std + 1e-12)


        data_mu = 0
        data_std = 0

    elif args.datatype == "NADC":
        data_infor = np.load(data_path, allow_pickle=True)
        data_end = data_infor['data']
        print(data_end.shape)
        labels = data_infor['label']
        four_tag = data_infor['four_tag']

        data_mu = np.mean(data_end, axis=0)
        data_std = np.std(data_end, axis=0)
        data_process = (data_end - data_mu) / (data_std + 1e-10)


        data_mu = 0
        data_std = 0


    elif args.datatype == "toy":
        data_label = np.load(data_path)
        data_process = data_label['x']
        labels = data_label['y']
        data_mu = 0
        data_std = 0

    else:
        print(args.datatype)
        raise NotImplementedError("not implemented the dataset")

    print('data_size=', data_process.shape)

    norma_idx = np.where(labels == 0)[0][:20000]
    # attack_idx = np.where(labels != 0)[0][:50]

    if args.datatype == "toy":
        attack_idx_array = np.where(labels != 0)[0]
    else:
        attack_idx = np.where(labels != 0)[0]
        attack_label_set = set(labels[attack_idx])
        attack_idx_lst = []
        for label in attack_label_set:
            attack_idx_lst.extend(list(np.where(labels == label)[0][:20]))

        attack_idx_array = np.array(attack_idx_lst)



    data_train = data_process[norma_idx]
    label_train = np.array(labels[norma_idx], dtype=int)

    data_test = data_process[attack_idx_array]
    label_test = np.array(labels[attack_idx_array], dtype=int)

    print('train shape=', data_train.shape, Counter(label_train))
    print('test shape=', data_test.shape, Counter(label_test))


    return data_train, label_train, data_test, label_test, data_mu, data_std





class Root_Cause_Analysis:

    def __init__(self, args, dim):
        self.datatype = args.datatype
        self.train_date = args.train_date
        self.device = args.device
        self.dim = dim
        self.exp_method = args.exp_method

        self.epochs = args.epochs
        self.lr = args.lr
        self.alpha = args.alpha
        self.eps = args.eps
        self.t1 = args.t1
        self.t2 = args.t2
        self.fea_t = args.fea_t



    def explain(self, explain_data, explain_label, proto_dict, data_train):

        proto_array = np.array([proto for proto in proto_dict[0]])


        explanation_lst = []
        rca_explain = RCA_Explain(epochs=self.epochs,
                              lr=self.lr,
                              alpha=self.alpha,
                              eps=self.eps,
                              t1=self.t1,
                              t2=self.t2,
                              dim=self.dim,
                              device=self.device,
                              fea_t = self.fea_t

                              )

        for i in tqdm(range(len(explain_data))):
            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]

            print('-------------------------')
            explanation = rca_explain.explain_on_nearest_proto(explain_instance, explain_instance_label, proto_array, data_train)
            explanation_lst.append(explanation)


        return explanation_lst



def MatrixEuclideanDistances(a, b):

    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()

    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

class RCA_Explain:

    def __init__(self, epochs, lr, alpha,  eps, t1, t2, dim, device, fea_t):

        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.t1 = t1
        self.t2 = t2
        self.device = device
        self.early_stop = True
        self.dim = dim
        self.fea_t = fea_t



    def concrete_transformation(self, p):

        reverse_theta = torch.ones_like(p) - p
        appro = torch.log(p + self.eps) - torch.log(reverse_theta + self.eps)

        logit = appro / self.t1
        return torch.sigmoid(logit)

    def results_print_on_nearest_proto(self, explain_instance, feature_importance, nearest_proto, epoch):


        imp_idx = np.where(feature_importance>=self.fea_t + self.eps)[0]

        print('Num_imp_fea=', len(imp_idx))

        replace = np.zeros(len(feature_importance))
        replace[imp_idx] = 1
        remain = np.abs(1 - replace)

        instances = np.array(nearest_proto.instances)

        # dist_r = euclidian_dist(instances, nearest_proto.center)
        # print(nearest_proto.r, euclidian_dist(explain_instance, nearest_proto.center), len(instances))
        # idx = np.where(dist_r > nearest_proto.r)[0]
        # print(idx, len(idx))


        # dist = euclidian_dist(instances, nearest_proto.center)
        # idx = np.where(dist <= nearest_proto.r)[0]
        # instances = instances[idx]
        data_train_new = replace * instances + remain * explain_instance
        dist = euclidian_dist(data_train_new, nearest_proto.center)
        in_circle = len(np.where(dist <= nearest_proto.r + self.eps)[0]) / len(instances)


        # print('fea_imp=', feature_importance)
        # print("dist=", dist, nearest_proto.r)

        data_train_weight_new = feature_importance * instances + (1 - feature_importance) * explain_instance
        dist_weight = euclidian_dist(data_train_weight_new, nearest_proto.center)
        in_circle_weight = len(np.where(dist_weight <= nearest_proto.r + self.eps)[0]) / len(instances)


        data_center_new = replace * nearest_proto.center + remain * explain_instance
        dist_centers = euclidian_dist(data_center_new, nearest_proto.center)
        if dist_centers <= nearest_proto.r + self.eps:
            in_circle_center = 1
        else:
            in_circle_center = 0

            # idx = np.where(dist > nearest_proto.r)[0]
            # # # print(len(instances), len(idx))
            # #
            # plot_data = np.vstack((instances, nearest_proto.center.reshape(1, -1), explain_instance.reshape(1, -1)))
            # plot_label = np.zeros(len(instances) + 2)
            # plot_label[-2] = 1
            # plot_label[-1] = 3
            # plot_label[idx] = 2
            # tsne_plot(plot_data, plot_label, path=f'../../figure/instances_plot_{str(epoch)}')


        return in_circle, in_circle_weight,in_circle_center


    def explain_on_nearest_proto(self, explain_instance, explain_label, proto_array, data_train):

        begin_time = time.time()
        center_array = np.array([proto.center for proto in proto_array])
        nearest_proto_idx = np.argsort(euclidian_dist(explain_instance, center_array))[0]
        nearest_proto = proto_array[nearest_proto_idx]
        nearest_proto_center = nearest_proto.center
        nearest_proto_r = nearest_proto.r
        nearest_proto_ss = nearest_proto.ss
        nearest_proto_size = nearest_proto.size
        instances_in_proto = np.array(nearest_proto.instances)



        dist = euclidian_dist(explain_instance, nearest_proto_center)[0]
        if dist <= nearest_proto_r:
            explain_instance_in_out_flag = "In the Prototype!!!!!!!!!!!!!!!!!"
        else:
            explain_instance_in_out_flag = "Out the Prototype"

        print('attack:', explain_label, explain_instance_in_out_flag, 'size=', nearest_proto_size)


        explain_instance = torch.from_numpy(explain_instance).type(torch.float).to(self.device)
        nearest_proto_center = torch.from_numpy(nearest_proto_center).type(torch.float).to(self.device)
        nearest_proto_ss = torch.from_numpy(nearest_proto_ss).type(torch.float).to(self.device)
        instances_in_proto = torch.from_numpy(instances_in_proto).type(torch.float).to(self.device)
        nearest_proto_r = torch.tensor(nearest_proto_r).type(torch.float).to(self.device)
        nearest_proto_size = torch.tensor(nearest_proto_size).type(torch.float).to(self.device)

        relu = nn.ReLU()
        w_fea = torch.ones(len(explain_instance)).type(torch.float).to(self.device)
        w_fea.requires_grad = True

        optimizer = torch.optim.Adam([w_fea], lr=self.lr)
        Last_Loss = np.Inf
        for epoch in range(self.epochs):
            w_fea_normalize = torch.sigmoid(w_fea)
            mask_fea = self.concrete_transformation(w_fea_normalize)
            mask_reverse = torch.ones_like(mask_fea) - mask_fea

            x_new = explain_instance * mask_reverse + mask_fea * nearest_proto_center
            dist = torch.sum((x_new - nearest_proto_center) ** 2)
            # compensation = torch.sum(mask_fea ** 2 * (nearest_proto_ss / nearest_proto_size - nearest_proto_center ** 2))
            compensation = 0
            instance_dist_tensor = torch.sqrt(dist + compensation) - nearest_proto_r + self.eps

            # x_new = explain_instance * mask_reverse + mask_fea * instances_in_proto
            # dist = torch.sum((x_new - nearest_proto_center) ** 2, dim=1)
            # instance_dist_tensor = torch.sqrt(dist) - nearest_proto_r + self.eps
            # Loss1 = torch.mean(relu(instance_dist_tensor))

            Loss1 = relu(instance_dist_tensor)
            Loss2 = torch.norm(mask_fea, p=1) / self.dim
            Loss = Loss1 + self.alpha * Loss2
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            if epoch % 100 == 0 or epoch == self.epochs-1:
                with torch.no_grad():
                    explain_instance_npy = explain_instance.cpu().data.numpy()

                    # actual_results = self.results_print_on_nearest_proto(explain_instance_npy,
                    #                                                     mask_fea.cpu().data.numpy(),
                    #                                                     nearest_proto, epoch,
                    #                                                     )
                print(f'Epoch:{epoch + 1:02}/{self.epochs:02}| '
                      f'Loss:{Loss:.8f}| '
                      f'Loss1:{Loss1:.8f}| ',
                      f'Loss2:{Loss2:.8f}| '
                      )
                      # actual_results)

                # idx_sort = np.argsort(-mask_fea.cpu().data.numpy())
                # print('mask_fea_idx_sort=', idx_sort)
                # print('mask_fea=', mask_fea.cpu().data.numpy())
                # print('relu_dist=', relu(instance_dist_tensor))
                # print('-------------------------------------')

        end_time = time.time()

        w_numpy = mask_fea.cpu().data.numpy()
        imp_idx = np.where(w_numpy >= self.fea_t + self.eps)[0]
        print(w_numpy, np.argsort(-w_numpy))

        if len(imp_idx) == 0:
            imp_idx = np.array([np.argmax(w_numpy)])

        print('imp_fea_num=', len(imp_idx))
        feature_importance = w_numpy

        print('Finish Interpretation after {} epoch'.format(epoch), '(Final loss: %.2f,' % Loss.item(),
              "Time elasped: %.2fs)" % (end_time - begin_time))
        return feature_importance


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(prog="arguments")
    arg_parser.add_argument('--datatype', type=str, help="Dataset type.", default='toy')
    arg_parser.add_argument('--train_date', type=str, help="train data date", default='1')
    arg_parser.add_argument('--device', type=str, help="Torch device.", default='cuda:0',
                            choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    arg_parser.add_argument('--exp_method', type=str, help="train data date", default='Clnb')
    arg_parser.add_argument('--tag', type=int, help="number", default=1)
    arg_parser.add_argument('--center_flag', type=str, help="center_or_alldata", default="True")
    arg_parser.add_argument('--ipdt', type=float, help="threshold of proto", default=1.0)
    arg_parser.add_argument('--fea_t', type=float, help="threshold of important_feature", default=0.5)

    arg_parser.add_argument('--epochs', type=int, help="epochs", default=500)
    arg_parser.add_argument('--lr', type=float, help="learning rate", default=0.01)
    arg_parser.add_argument('--alpha', type=float, help="loss_ratio1", default=1e-4)
    arg_parser.add_argument('--eps', type=float, help="error rate", default=1e-6)
    arg_parser.add_argument('--t1', type=float, help="temperature_1", default=0.1)
    arg_parser.add_argument('--t2', type=float, help="temperature_2", default=0.001)

    args = arg_parser.parse_args()


    data_train, label_train, data_test, label_test, data_mu, data_std = read_train_test_data(args)
    dim = data_train.shape[1]
    time1 = time.time()
    proto = Prototypes(args)

    proto_path = f'../../protos_sample/protos_{args.datatype}_{args.train_date}_{args.ipdt}'+'.npy'

    if not os.path.exists(proto_path):
        proto_dict = proto.proto_fit(data_train, label_train)
        np.save(proto_path, proto_dict)
    else:
        proto_dict = np.load(proto_path, allow_pickle=True).item()


    proto_array = np.array([proto for proto in proto_dict[0]])
    center_array = np.array([proto.center for proto in proto_dict[0]])
    r_array = np.array([proto.r for proto in proto_dict[0]])
    print('Norm_proto_num=', len(proto_dict[0]))

    outside_idx = []
    for i in range(len(data_test)):
        dist = euclidian_dist(data_test[i], center_array)
        cover_idx = np.where(dist <= r_array + args.eps)[0]
        if len(cover_idx) == 0:
            outside_idx.append(i)
    outside_idx_array = np.array(outside_idx, dtype=int)

    data_test = data_test[outside_idx_array]
    label_test = label_test[outside_idx_array]



    rca = Root_Cause_Analysis(args, dim)
    f_imp_lst = rca.explain(data_test, label_test, proto_dict, data_train)
    print(len(proto_dict[0]))

    time2 = time.time()

    in_circle_lst = []
    in_circle_weight_lst = []
    in_circle_center_lst = []

    imp_fea_num = 0

    for i in range(len(f_imp_lst)):

        feature_importance = f_imp_lst[i]
        imp_idx = np.where(feature_importance >= args.fea_t + args.eps)[0]
        if len(imp_idx) == 0:
            imp_idx = np.array([np.argmax(feature_importance)])
        imp_fea_num += len(imp_idx)
        replace = np.zeros(len(feature_importance))
        replace[imp_idx] = 1
        remain = np.abs(1 - replace)

        min_idx = np.argmin(euclidian_dist(data_test[i], center_array))
        nearest_proto = proto_array[min_idx]
        instances_in_proto = np.array(nearest_proto.instances)

        data_train_new = replace * instances_in_proto + remain * data_test[i]
        dist = euclidian_dist(data_train_new, nearest_proto.center)
        in_circle = len(np.where(dist <= nearest_proto.r + args.eps)[0]) / len(data_train_new)

        data_train_weight_new = feature_importance * instances_in_proto + (1 - feature_importance) * data_test[i]
        dist_weight = euclidian_dist(data_train_weight_new, nearest_proto.center)
        in_circle_weight = len(np.where(dist_weight <= nearest_proto.r + args.eps)[0]) / len(instances_in_proto)

        data_center_new = replace * nearest_proto.center + remain * data_test[i]
        dist_centers = euclidian_dist(data_center_new, nearest_proto.center)
        if dist_centers <= nearest_proto.r + args.eps:
            in_circle_center = 1
        else:
            in_circle_center = 0

        in_circle_lst.append(in_circle)
        in_circle_weight_lst.append(in_circle_weight)
        in_circle_center_lst.append(in_circle_center)



    print(f'exp_method={args.exp_method}, time=', "%.2f" % (time2 - time1))
    print('in_circle_num=', np.mean(in_circle_lst))
    print('in_circle_center_num=', np.mean(in_circle_center_lst))
    print('in_circle_weight_num=', np.mean(in_circle_weight_lst))
    print('avg_imp_fea_num=', round(imp_fea_num / len(f_imp_lst), 2))