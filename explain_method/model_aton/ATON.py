"""
This script implements an outlier interpretation method of the following paper:
"Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network". in WWW'21.
@ Author: Hongzuo Xu
@ email: hongzuo.xu@gmail.com or leogarcia@126.com or xuhongzuo13@nudt.edu.cn
"""

from pyod.models import lscp
import numpy as np
import time, math
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from tqdm import tqdm
from explain_method.model_aton.utils import EarlyStopping, min_max_normalize

from explain_method.model_aton.datasets import MyHardSingleTripletSelector
from explain_method.model_aton.datasets import SingleTripletDataset
from explain_method.model_aton.networks import ATONnet, AttentionNet
from explain_method.model_aton.networks import MyLoss


class ATON:
    def __init__(self, nbrs_num=30, rand_num=30, alpha1=0.8, alpha2=0.2,
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=2.,
                 verbose=True, gpu=True):
        self.verbose = verbose

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        # a list of normal nbr of each anomaly
        self.normal_nbr_indices = []

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda and gpu else "cpu")
        if cuda:
            torch.cuda.set_device(0)
        # print("device:", self.device)

        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_linear = n_linear
        self.margin = margin
        return

    def fit(self, x, y):
        # print("Begin Explain")
        device = self.device

        self.dim = x.shape[1]
        x = min_max_normalize(x)
        self.ano_idx = np.where(y != 0)[0]

        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)
        self.prepare_nbrs()
        # train model for each anomaly
        attn_lst, W_lst = [], []


        for ii in range(len(self.ano_idx)):
            idx = self.ano_idx[ii]
            s_t = time.time()
            attn, W = self.interpret_ano(ii)
            attn_lst.append(attn)
            W_lst.append(W)



        fea_weight_lst = []
        for ii, idx in enumerate(self.ano_idx):
            attn, w = attn_lst[ii], W_lst[ii]
            fea_weight = np.zeros(self.dim)
            # attention (linear space) + w --> feature weight (original space)
            for j in range(len(attn)):
                fea_weight += attn[j] * abs(w[j])
            fea_weight_lst.append(fea_weight)

        return fea_weight_lst

    def interpret_ano(self, ii):
        idx = self.ano_idx[ii]
        device = self.device
        dim = self.dim

        nbr_indices = self.normal_nbr_indices[ii]
        data_loader, test_loader = self.prepare_triplets(idx, nbr_indices)
        n_linear = self.n_linear
        attn_net = AttentionNet(in_feature=3 * n_linear, n_hidden=int(1.5 * n_linear), out_feature=n_linear)
        model = ATONnet(attn_net=attn_net, n_feature=dim, n_linear=n_linear)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = MyLoss(alpha1=self.alpha1, alpha2=self.alpha2, margin=self.margin)

        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        early_stp = EarlyStopping(patience=3, verbose=False)

        for epoch in range(self.n_epoch):
            model.train()
            total_loss = 0
            total_dis = 0
            es_time = time.time()

            batch_cnt = 0
            # random, neighbor, anomaly
            for anchor, pos, neg in data_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                embed_anchor, embed_pos, embed_neg, attn, dis = model(anchor, pos, neg)
                loss = criterion(embed_anchor, embed_pos, embed_neg, dis)

                total_loss += loss
                total_dis += dis.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_cnt += 1

            train_loss = total_loss / batch_cnt
            est = time.time() - es_time

            if self.verbose and (epoch + 1) % 1 == 0:
                message = 'Epoch: [{:02}/{:02}]  loss: {:.4f} Time: {:.2f}s'.format(epoch + 1, self.n_epoch,
                                                                                    train_loss, est)
                print(message)
            scheduler.step()

            early_stp(train_loss, model)
            if early_stp.early_stop:
                model.load_state_dict(torch.load(early_stp.path))
                if self.verbose:
                    print("early stopping")
                break

        # distill W and attn from network
        for anchor, pos, neg in test_loader:
            model.eval()
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            _, _, _, attn, _ = model(anchor, pos, neg)

        attn_avg = torch.mean(attn, dim=0)
        attn_avg = attn_avg.data.cpu().numpy()
        W = model.linear.weight.data.cpu().numpy()
        return attn_avg, W

    def prepare_triplets(self, idx, nbr_indices):
        x = self.x
        y = self.y
        selector = MyHardSingleTripletSelector(nbrs_num=self.nbrs_num, rand_num=self.rand_num,
                                               nbr_indices=nbr_indices)
        dataset = SingleTripletDataset(idx, x, y, triplets_selector=selector)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader

    def prepare_nbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        x_anom = x[anom_idx]
        noml_idx = np.where(y == 0)[0]
        x_noml = x[noml_idx]
        n_neighbors = self.nbrs_num

        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(x_noml)
        tmp_indices = nbrs_local.kneighbors(x_anom)[1]

        for idx in tmp_indices:
            nbr_indices = noml_idx[idx]
            self.normal_nbr_indices.append(nbr_indices)
        return
