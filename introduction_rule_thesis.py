import os
from explain_intro.model_deepaid.Model_DeepAID import Model_DeepAID
from explain_intro.model_deepaid.Deepaid_rule import Deepaid_rule_generation
from explain_intro.model_aton.Model_Aton import Model_Aton
from explain_intro.model_shap.Model_Shap import Model_Shap
from explain_intro.model_enids.ENIDS import ENIDS
from explain_intro.model_enids.Enids_rule import Enids_rule_generation
from explain_intro.model_explainer.Explainer import Explainer
from explain_intro.model_skoperules.Model_SkopeRules import Skoprules
from explain_intro.model_anchor.Anchor import Anchor_rule
from evaluation.evaluate_tools import evaluate_acc_f1, roc_auc, pr
from scipy.stats import gaussian_kde
import numpy as np
import time
from joblib import Parallel, delayed
from utils import read_train_test_data_stream, get_feature_names, \
    rule_process
from evaluation.feature_rule_evaluate_all import stream_rule_evaluate, stream_feature_evaluate_ids
import torch
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
from compara_config import get_comparative_parser
from bisect import bisect_left

def plotMatrixPoint(normal_data, attack_data, mu, std):
    attack_data_1 = attack_data[14]
    attack_data_2 = attack_data[30]

    color1 = [64 / 255, 116 / 255, 52 / 255]
    color2 = [229 / 255, 131 / 255, 8 / 255]

    fig = plt.figure(1, figsize=(12, 9))
    # prec_on_alldata = np.array([0, 0.665, 0.683, 0.740])
    prec_on_prototypes = np.array([0.827, 0.598, 0.525, 0.377, 0.43])
    jac_on_prototypes = np.array([0.706, 0.444, 0.363, 0.235, 0.28])

    x = list(np.array(range(5)))
    width = 0.25
    x[0] = x[0] - 0.05
    for i in range(len(x)):
        x[i] = x[i] - width

    plt.bar(x, prec_on_prototypes, width=width, label='精确率', tick_label=[r"$\bf{DPAI}$", 'DeepAID', 'ATON', 'SHAP', 'ACE-KL'], color= color1)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, jac_on_prototypes, width=width, label='Jaccard相似系数', color= color2)


    fontdict_label_ch = {'weight': 'bold', 'size': 25, 'family': 'SimHei'}
    plt.ylabel("告警解释效果", fontdict=fontdict_label_ch)
    plt.xticks(size=20, rotation='30')
    plt.yticks(np.arange(0.0,1.01,0.2), size=20)

    fontdict_title_ch = {'weight': 'bold', 'size': 25, 'family': 'SimHei'}
    # plt.title("特征解释评估", fontdict=fontdict_title_ch, x=0.5,y=1.025)

    fontdict_leg_ch = {'weight': 'bold', 'size': 20, 'family': 'SimHei'}
    plt.legend(ncol=2, loc='upper center',prop=fontdict_leg_ch)

    # -------------------------------------------

    # axes2 = plt.subplot(223)  # figure1的子图1为axes1
    # jac_on_alldata = np.array([0, 0.512, 0.529, 0.594])
    # jac_on_prototypes = np.array([0.706, 0.444, 0.363, 0.235])
    # x = list(np.array(range(4)))
    # width = 0.25
    # x[0] = x[0] - 0.05
    # for i in range(len(x)):
    #     x[i] = x[i] - width
    #
    # plt.bar(x, jac_on_alldata, width=width, fc=color1, tick_label=[r"$\bf{Ours}$", 'DeepAID-f', 'ATON', 'SHAP'], label='on All Data')
    #
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    #
    # plt.bar(x, jac_on_prototypes, width=width, fc=color2, label='on Prototypes',)
    #
    # plt.ylabel("Jaccard", fontsize=20)
    # plt.xticks(size=20, rotation='30')
    # plt.yticks(size=20)
    # plt.ylim(0.0, 1.0)
    # plt.title("(b) Jaccard of Feature Explanations", size=25)
    # plt.legend(ncol=2, loc='upper center',fontsize=20)


    # attack_data_1 = attack_data[14]
    # attack_data_2 = attack_data[30]
    #
    # color1 = [64 / 255, 116 / 255, 52 / 255]
    # color2 = [229 / 255, 131 / 255, 8 / 255]
    #
    # fig = plt.figure(1, figsize=(20, 7))
    # axes1 = plt.subplot(121)  # figure1的子图1为axes1
    # prec_on_alldata = np.array([0, 0.665, 0.683, 0.740])
    # prec_on_prototypes = np.array([0.827, 0.598, 0.525, 0.377])
    #
    #
    # x = list(np.array(range(4)))
    # width = 0.25
    # x[0] = x[0] - 0.05
    # for i in range(len(x)):
    #     x[i] = x[i] - width
    #
    # plt.bar(x, prec_on_alldata, width=width, label='on All Data', fc=color1,
    #         tick_label=[r"$\bf{Ours}$", 'DeepAID-f', 'ATON', 'SHAP'])
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, prec_on_prototypes, width=width, label='on Prototypes', fc=color2)
    #
    #
    # plt.ylabel("Precision", fontsize=20)
    # plt.xticks(size=20, rotation='30')
    # plt.yticks(size=20)
    # plt.ylim(0.0, 1.0)
    # plt.title("(a) Precsion of Feature Explanations", size=25)
    # plt.legend(ncol=2, loc='upper center', fontsize=20)
    #
    # # -------------------------------------------
    #
    # # axes2 = plt.subplot(223)  # figure1的子图1为axes1
    # # jac_on_alldata = np.array([0, 0.512, 0.529, 0.594])
    # # jac_on_prototypes = np.array([0.706, 0.444, 0.363, 0.235])
    # # x = list(np.array(range(4)))
    # # width = 0.25
    # # x[0] = x[0] - 0.05
    # # for i in range(len(x)):
    # #     x[i] = x[i] - width
    # #
    # # plt.bar(x, jac_on_alldata, width=width, fc=color1, tick_label=[r"$\bf{Ours}$", 'DeepAID-f', 'ATON', 'SHAP'], label='on All Data')
    # #
    # # for i in range(len(x)):
    # #     x[i] = x[i] + width
    # #
    # # plt.bar(x, jac_on_prototypes, width=width, fc=color2, label='on Prototypes',)
    # #
    # # plt.ylabel("Jaccard", fontsize=20)
    # # plt.xticks(size=20, rotation='30')
    # # plt.yticks(size=20)
    # # plt.ylim(0.0, 1.0)
    # # plt.title("(b) Jaccard of Feature Explanations", size=25)
    # # plt.legend(ncol=2, loc='upper center',fontsize=20)



# ----------------------------------------------------------------------
#     hist, bins = np.histogram(normal_data, bins=100)
#     hist = np.sqrt(hist)
#     hist_score = np.zeros(len(normal_data))
#     for i in range(len(normal_data)):
#         idx = bisect_left(bins, normal_data[i]) - 1
#         hist_score[i] = hist[idx]
#     # print(np.max(hist_score))
#     # exit()
#     # axes3 = plt.subplot(222)  # figure1的子图1为axes1
#     # plt.hist(normal_data, bins=50, color='b')
#     # plt.ylabel('Number of Normal Traffic',size=20)
#     # plt.xlabel('Feature values',size=20)
#     # plt.yticks(size=20)
#     # plt.xticks(size=20)
#     # plt.yticks(np.linspace(0.0, 40000, 5),size=20)
#     # plt.title('(c) Histogram of Normal Traffic on $cnt\_dst\_conn$', size=25)
#
#     # hist_score[0] = 300
#     # ----------------------------------------------------------------------
#     axes4 = plt.subplot(122)  # figure1的子图1为axes1
#     color3 = [0/255,90/255,171/255]
#     color4 = [29/255,191/255,151/255]
#     color5 = [87/255,96/255,105/255]
#
#     line_weight = 5
#     solid_weight = 2
#
#     # y = -1
#     # im = axes4.scatter(normal_data, y * np.ones(len(normal_data)), s=30, c=hist_score, cmap="inferno_r")
#     # im = axes4.scatter(normal_data, y * np.ones(len(normal_data)), s=30, c=hist_score, cmap="binary")
#     # for i in range(len(attack_data)):
#     #     axes4.scatter(attack_data[i], (-1) * np.ones(1), s=30, c='r', marker='o')  # scatter函数只支持array类型数据
#
#     # axes4.scatter(attack_data_1, y * np.ones(1), s=80, c='purple', marker='^')  # scatter函数只支持array类型数据
#     # intervals = [(int(3.14*std+mu), np.max(normal_data)), (int(4.936*std+mu), int(6.176*std+mu)), (int(5.872*std+mu), np.max(normal_data))]
#     # axes4.hlines(y=y + 0.2, xmin=intervals[0][0], xmax=intervals[0][1]-0.5, colors=color3, label=r'$\bf{Ours}$: 0.99, 0.98', lw=line_weight)
#     # axes4.vlines(x=intervals[0][0], ymin=y+0.05, ymax=y + 0.2, colors=color3, lw=solid_weight, ls="--")
#     # axes4.vlines(x=intervals[0][1]-0.5, ymin=y+0.05, ymax=y + 0.2, colors=color3, lw=solid_weight, ls="--")
#     #
#     # axes4.hlines(y=y + 0.4, xmin=intervals[1][0], xmax=intervals[1][1], colors=color4, label='Explainer: 0.55, 0.53',lw=line_weight)
#     # axes4.vlines(x=intervals[1][0], ymin=y + 0.05, ymax=y + 0.4, colors=color4, lw=solid_weight, ls="--")
#     # axes4.vlines(x=intervals[1][1], ymin=y + 0.05, ymax=y + 0.4, colors=color4, lw=solid_weight, ls="--")
#     #
#     #
#     # axes4.hlines(y=y + 0.6, xmin=intervals[2][0], xmax=intervals[2][1], colors=color5, label='DeepAID-r: 0.92, 0.77',lw=line_weight)
#     # axes4.vlines(x=intervals[2][0], ymin=y + 0.05, ymax=y + 0.6, colors=color5, lw=solid_weight, ls="--")
#     # axes4.vlines(x=intervals[2][1], ymin=y + 0.05, ymax=y + 0.6, colors=color5, lw=solid_weight, ls="--")
#
#     y_normal = 0
#     # axes4.scatter(normal_data, y * np.ones(len(normal_data)), s=30, c=hist_score, cmap="binary")
#     im_normal = axes4.scatter(normal_data, y_normal * np.ones(len(normal_data)), s=50, c=hist_score, cmap="binary")
#
#
#     # for i in range(len(attack_data)):
#     #     axes4.scatter(attack_data[i], y * np.ones(1), s=50, c='r', marker='o')  # scatter函数只支持array类型数据
#
#
#     y = 0.2
#     hist_attack, bins_attack = np.histogram(attack_data, bins=20)
#     hist_attack = np.sqrt(hist_attack)
#     hist_score_attack = np.zeros(len(attack_data))
#     for i in range(len(attack_data)):
#         idx = bisect_left(bins_attack, attack_data[i]) - 1
#         hist_score_attack[i] = hist_attack[idx]
#
#
#     im_attack = axes4.scatter(attack_data, y * np.ones(len(attack_data)), s=50, c=hist_score_attack, cmap="Reds")
#
#     axes4.scatter(attack_data_2, y * np.ones(1), s=100, c='magenta', marker='^')  # scatter函数只支持array类型数据
#     intervals = [(int(4.323 * std + mu), np.max(normal_data)), (int(7.592 * std + mu), int(8. * std + mu)),
#                  (int(7.804 * std + mu), np.max(normal_data))]
#     axes4.hlines(y=y + 0.2, xmin=intervals[0][0], xmax=intervals[0][1]-0.5, colors=color3, label=r'$\bf{Ours:0.98}$', lw=line_weight)
#     axes4.vlines(x=intervals[0][0], ymin=y + 0.01, ymax=y + 0.2, colors=color3, lw=solid_weight, ls="--")
#     axes4.vlines(x=intervals[0][1]-0.5, ymin=y + 0.01, ymax=y + 0.2, colors=color3, lw=solid_weight, ls="--")
#
#     axes4.hlines(y=y + 0.4, xmin=intervals[1][0], xmax=intervals[1][1], colors=color4, label='Explainer:0.53', lw=line_weight, ls='dotted')
#     axes4.vlines(x=intervals[1][0], ymin=y + 0.01, ymax=y + 0.4, colors=color4, lw=solid_weight, ls="--")
#     axes4.vlines(x=intervals[1][1], ymin=y + 0.01, ymax=y + 0.4, colors=color4, lw=solid_weight, ls="--")
#
#     axes4.hlines(y=y + 0.6, xmin=intervals[2][0], xmax=intervals[2][1], colors=color5, label='DeepAID-r:0.77', lw=line_weight, ls='dashdot')
#     axes4.vlines(x=intervals[2][0], ymin=y + 0.01, ymax=y + 0.6, colors=color5, lw=solid_weight, ls="--")
#     axes4.vlines(x=intervals[2][1], ymin=y + 0.01, ymax=y + 0.6, colors=color5, lw=solid_weight, ls="--")
#
#     axes4.legend(ncol=1, fontsize=16, loc='upper left', handlelength=5)
#
#     # position_normal = fig.add_axes([0.5,0.05,0.7,0.03])
#     cbar_normal = plt.colorbar(im_normal,  orientation='horizontal')
#     ticks = [1,25,50,75,100]
#     ticklabels = [tick**2 for tick in ticks]
#     cbar_normal.set_ticks(ticks)
#     cbar_normal.set_ticklabels(ticklabels)
#     cax = plt.gcf().axes[-1]
#     cax.tick_params(labelsize=20)
#
#     fontdict_title_ch = {'weight': 'bold', 'size': 20, 'family': 'SimHei'}
#     cbar_normal.set_label('正常流量的数量', fontdict=fontdict_title_ch)
#
#     # position_attack = fig.add_axes([axes4.get_position(), 0.25, axes4.get_position()+0.4, 0.03])
#     cbar_attack = plt.colorbar(im_attack, orientation='horizontal')
#     ticks = [1, 2, 3, 4, 5]
#     ticklabels = [tick ** 2 for tick in ticks]
#     cbar_attack.set_ticks(ticks)
#     cbar_attack.set_ticklabels(ticklabels)
#     cax_attack = plt.gcf().axes[-1]
#     cax_attack.tick_params(labelsize=20)
#
#     fontdict_title_ch = {'weight': 'bold', 'size': 20, 'family': 'SimHei'}
#     cbar_attack.set_label('攻击的数量', fontdict=fontdict_title_ch)
#
#     ticks = [0, 0.2, 1]
#
#     labels = ['正常\n流量', '攻击','']
#     axes4.set_yticks(ticks)
#     fontdict_title_ch = {'weight': 'bold', 'size': 15, 'family': 'SimHei'}
#     axes4.set_yticklabels(labels, fontdict_title_ch)
#
#     plt.yticks(size=20)
#     plt.xticks(size=20)
#
#     fontdict_title_ch = {'weight': 'bold', 'size': 20, 'family': 'SimHei'}
#     plt.xlabel('特征值', fontdict=fontdict_title_ch)
#     # axes4.get_yaxis().set_visible(False)
#
#     fontdict_title_ch = {'weight': 'bold', 'size': 25, 'family': 'SimHei'}
#     plt.title('(b) 规则解释评估', fontdict=fontdict_title_ch, x=0.5,y=1.05)
#     plt.subplots_adjust(wspace=0.2)

    plt.savefig('./figure/intro_figure', bbox_inches='tight', dpi=512)



def get_proto_param(datatype, train_date):
    proto_size_t = 10
    if datatype == 'NADC':
        ipdt = 0.5
        batch_size = 50000
        if train_date == '1203':
            stream_size = 750000
        elif train_date == '1210':
            stream_size = 1150000
        elif train_date == '1216':
            stream_size = 1800000
        else:
            raise NotImplementedError

    elif datatype == 'IDS2017':
        ipdt = 1.0
        batch_size = 20000
        if train_date == 'Tuesday':
            stream_size = 460000
        elif train_date == 'Wednesday':
            stream_size = 100000
        elif train_date == 'Thursday':
            stream_size = 500000
        elif train_date == 'Friday':
            stream_size = 240000
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return stream_size, batch_size, ipdt, proto_size_t

class Config:

    def __init__(self, args):
        self.a_ipdt = args.a_ipdt
        self.epochs = args.epochs
        self.lr = args.lr
        self.alpha = args.alpha
        self.t1 = args.t1
        self.t2 = args.t2
        self.store_w = args.store_w

        self.loss_guided = args.loss_guided
        self.keypoints = args.keypoints
        self.percentile = args.percentile
        self.tolerance = args.tolerance

        stream_size, batch_size, ipdt, proto_size_t = get_proto_param(args.datatype, args.train_date)

        self.stream_size = stream_size
        self.batch_size = batch_size
        self.ipdt = ipdt
        self.proto_size_t = proto_size_t



def main(args):

    print("This is", args.exp_method, args.explain_flag, args.datatype, args.train_date, "!!!!!")

    config = Config(args)

    print('Enids Param=')
    print(
        f'a_ipdt={config.a_ipdt}, '
        f'epochs={config.epochs}, '
        f'lr={config.lr}, '
        f'alpha={config.alpha}, '
        f't1={config.t1}, '
        f't2={config.t2}, '
        f'store_w={config.store_w}, '
        f'loss_guided={config.loss_guided}, '
        f'keypoints={config.keypoints}, '
        f'percentile={config.percentile}, '
        f'tolerance={config.tolerance}'
    )

    # 1.Load Data and Label
    feature_names = get_feature_names(args.datatype)
    data_use, label_use, data_mu, data_std, attack_dict = read_train_test_data_stream(args, config)




    attack_label = args.attack_label

    original_feature_names = ["time", "src", "dst", "spt", "dpt", "duration", "out_bytes", "in_bytes", "proto", "app",
         "cnt_dst", "cnt_src", "cnt_serv_src", "cnt_serv_dst", "cnt_dst_slow", "cnt_src_slow",
         "cnt_serv_src_slow", "cnt_serv_dst_slow", "cnt_dst_conn", "cnt_src_conn", "cnt_serv_src_conn",
         "cnt_serv_dst_conn"]

    dim = data_use.shape[1]

    feature_num_name = f'Enids_{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}' \
                       f'_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
    feature_num_path = os.path.join('.', 'feature_num_intro')

    proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{config.stream_size}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    attacks_idx = np.where(label_use == attack_label)[0][50:50+args.attack_upper_num]
    # attacks_idx = np.where(label_use == attack_label)[0]

    attacks_data = data_use[attacks_idx]
    attacks_label = label_use[attacks_idx]







    if args.exp_method == 'Enids':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', args.exp_method)

        rule_path = os.path.join('.', 'rule_intro', 'Enids_r')
        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_time'

    elif args.exp_method == 'Deepaid':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', args.exp_method)

        rule_path = os.path.join('.', 'rule_intro', 'Deepaid_r')
        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_time'

    elif args.exp_method == 'Aton':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', args.exp_method)

    elif args.exp_method == 'Shap':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', args.exp_method)
    elif args.exp_method == 'Lime':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', args.exp_method)

    elif args.exp_method == 'Explainer':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', 'Enids')

        rule_path = os.path.join('.', 'rule_intro', 'Explainer')

        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_time'

    elif args.exp_method == 'Skrules':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', 'Enids')

        rule_path = os.path.join('.', 'rule_intro', 'Skrules')

        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_time'




    elif args.exp_method == 'Anchor':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}'
        feature_weight_path = os.path.join('.', 'feature_weight_intro', 'Enids')

        rule_path = os.path.join('.', 'feature_weight_intro', 'Anchor')

        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_time'



    else:
        raise NotImplementedError


    rule_feature_save_file = os.path.join(rule_path, rule_feature_name)
    rule_ineq_save_file = os.path.join(rule_path, rule_inequality_name)
    rule_threshold_save_file = os.path.join(rule_path, rule_threshold_name)
    rule_time_save_file = os.path.join(rule_path, rule_time_name)

    explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
    explained_data = attacks_data[explained_idx_in_attacks_array]
    explained_label = attacks_label[explained_idx_in_attacks_array]
    explained_idx = attacks_idx[explained_idx_in_attacks_array]



    attack_locate_idx = 0
    batch_idx = int(explained_idx[attack_locate_idx] / config.batch_size) - 1
    proto_name = f'batchidx_{batch_idx}.npy'
    normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
    idx_lst = sum([proto.data_idx_lst for proto in normal_protos], [])
    idx_array = np.array(idx_lst)
    normal_data_all = data_use[idx_array]
    normal_data = normal_data_all[:, 12]


    normal_data_original_on_feature = normal_data * data_std[12] + data_mu[12]
    explain_data_original_on_feature = explained_data[:, 12] * data_std[12] + data_mu[12]

    # calculate f1
    all_data_dimension = np.array(list(normal_data)+list(explained_data[:, 12]))
    print(all_data_dimension.shape)

    intro_label = np.zeros(len(all_data_dimension))
    intro_label[-len(explained_data):] = 1

    print('normal:', len(normal_data), 'attack:', len(explained_data))
    our_pred_1 = np.zeros(len(all_data_dimension))
    explainer_pred_1 = np.zeros(len(all_data_dimension))
    deepaid_pred_1 = np.zeros(len(all_data_dimension))
    for i in range(len(all_data_dimension)):
        if all_data_dimension[i] >= 3.14:
            our_pred_1[i] = 1
        if all_data_dimension[i] >= 4.936 and all_data_dimension[i] <= 6.176:
            explainer_pred_1[i] = 1
        if all_data_dimension[i] >= 5.872:
            deepaid_pred_1[i] = 1
    acc, precision, recall, f1_our_1, cm = evaluate_acc_f1(intro_label, our_pred_1)
    acc, precision, recall, f1_explainer_1, cm = evaluate_acc_f1(intro_label, explainer_pred_1)
    acc, precision, recall, f1_deepaid_1, cm = evaluate_acc_f1(intro_label, deepaid_pred_1)
    roc_our_1 = roc_auc(intro_label, our_pred_1)
    roc_explainer_1 = roc_auc(intro_label, explainer_pred_1)
    roc_deepaid_1 = roc_auc(intro_label, deepaid_pred_1)
    print('case 1----------------------------')
    print(f1_our_1, f1_explainer_1, f1_deepaid_1)
    print(roc_our_1, roc_explainer_1, roc_deepaid_1)
    print('pred_attack=', np.sum(our_pred_1))

    our_pred_2 = np.zeros(len(all_data_dimension))
    explainer_pred_2 = np.zeros(len(all_data_dimension))
    deepaid_pred_2 = np.zeros(len(all_data_dimension))
    for i in range(len(all_data_dimension)):
        if all_data_dimension[i] >= 4.323:
            our_pred_2[i] = 1
        if all_data_dimension[i] >= 7.592 and all_data_dimension[i] <= 8:
            explainer_pred_2[i] = 1
        if all_data_dimension[i] >= 7.704:
            deepaid_pred_2[i] = 1
    acc, precision, recall, f1_our_2, cm = evaluate_acc_f1(intro_label, our_pred_2)
    acc, precision, recall, f1_explainer_2, cm = evaluate_acc_f1(intro_label, explainer_pred_2)
    acc, precision, recall, f1_deepaid_2, cm = evaluate_acc_f1(intro_label, deepaid_pred_2)
    roc_our_2 = roc_auc(intro_label, our_pred_2)
    roc_explainer_2 = roc_auc(intro_label, explainer_pred_2)
    roc_deepaid_2 = roc_auc(intro_label, deepaid_pred_2)
    print('case 2----------------------------')
    print(f1_our_2, f1_explainer_2, f1_deepaid_2)
    print(roc_our_2, roc_explainer_2, roc_deepaid_2)





    plotMatrixPoint(normal_data_original_on_feature,
                    explain_data_original_on_feature,
                    data_mu[12],
                    data_std[12])


    exit()






    if not os.path.exists(rule_feature_save_file+'.npy'):

        if not os.path.exists(rule_path):
            os.makedirs(rule_path)

        if args.exp_method == 'Enids':
            f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
            LGP_idx = np.load(feature_weight_path + '/' + feature_weight_name + '_lgp.npy', allow_pickle=True)

            if args.exp_method == 'Enids':
                fea_num_dict = {}
                feat = args.feat
                imp_fea_num_lst = []
                for i in range(len(f_imp_array)):
                    feature_importance = f_imp_array[i]
                    imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                    if len(imp_idx) == 0:
                        imp_idx = np.array([np.argmax(feature_importance)])
                    imp_fea_num_lst.append(len(imp_idx))
                fea_num_dict[feat] = imp_fea_num_lst

            else:
                enids_feature_path = os.path.join('.', 'feature_weight_stream', 'Enids')
                enids_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
                enids_feature_array = np.load(enids_feature_path + '/' + enids_feature_name + '.npy',
                                              allow_pickle=True)
                fea_num_dict = {}
                feat = args.feat
                imp_fea_num_lst = []
                for i in range(len(enids_feature_array)):
                    feature_importance = enids_feature_array[i]
                    imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                    if len(imp_idx) == 0:
                        imp_idx = np.array([np.argmax(feature_importance)])
                    imp_fea_num_lst.append(len(imp_idx))
                fea_num_dict[feat] = imp_fea_num_lst


            feature_lst_explain = [[12] for i in range(len(explained_data))]

            time1 = time.time()
            feature_lst_explain, inequality_lst_explain, threshold_lst_explain = Enids_rule_generation(feature_lst_explain,
                                                                                                       LGP_idx,
                                                                                                       explained_data,
                                                                                                       explained_label,
                                                                                                       explained_idx,
                                                                                                       proto_path,
                                                                                                       config.batch_size,
                                                                                                       config.loss_guided,
                                                                                                       config.keypoints,
                                                                                                       config.percentile,
                                                                                                       config.tolerance,
                                                                                                       args.eps)
            time2 = time.time()




        elif args.exp_method == 'Deepaid':
            f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
            enids_rule_path = os.path.join('.', 'feature_weight_intro', 'Enids_r')
            enids_rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
            enids_rule_feature_array = np.load(enids_rule_path + '/' + enids_rule_feature_name + '.npy', allow_pickle=True)
            rule_num = [len(feature_lst) for feature_lst in enids_rule_feature_array]
            # rule_num = [2 for feature_lst in enids_rule_feature_array]
            time1 = time.time()
            feature_lst_explain, inequality_lst_explain, threshold_lst_explain = Deepaid_rule_generation(f_imp_array, rule_num, explained_data)
            time2 = time.time()


        elif args.exp_method == 'Explainer':

            time1 = time.time()
            explainer = Explainer(args, config)

            print(explained_data.shape)
            feature_lst_explain, inequality_lst_explain, threshold_lst_explain, rule_num_lst = \
                explainer.fit(explained_data[:,12].reshape(-1,1), explained_label, explained_idx, proto_path, data_use[:,12].reshape(-1,1))

            time2 = time.time()




        elif args.exp_method == 'Anchor':
            time1 = time.time()
            enids_rule_path = os.path.join('.', 'feature_weight_intro', 'Enids_r')
            enids_rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_attacklabel_{args.attack_label}_train_all_data_{args.train_all_data}_attack_upper_num_{args.attack_upper_num}_f'
            enids_rule_feature_array = np.load(enids_rule_path + '/' + enids_rule_feature_name + '.npy',
                                               allow_pickle=True)
            rule_num = [len(feature_lst) for feature_lst in enids_rule_feature_array]


            clf = Anchor_rule(args, feature_names, config)
            feature_lst_explain, inequality_lst_explain, threshold_lst_explain = clf.fit(explained_data[:,12],
                                                                                             explained_label,
                                                                                             explained_idx,
                                                                                             proto_path,
                                                                                         rule_num,
                                                                                             data_use)
            time2 = time.time()



        else:
            raise NotImplementedError

        feature_lst_explain, inequality_lst_explain, threshold_lst_explain = \
            rule_process(feature_lst_explain, inequality_lst_explain, threshold_lst_explain)




        time_explain = time2 - time1



        np.save(rule_feature_save_file, np.array(feature_lst_explain))
        np.save(rule_ineq_save_file, np.array(inequality_lst_explain))
        np.save(rule_threshold_save_file, np.array(threshold_lst_explain))
        np.save(rule_time_save_file, round(time_explain, 3))

    else:

        feature_lst_explain = np.load(rule_feature_save_file+'.npy', allow_pickle=True)
        inequality_lst_explain = np.load(rule_ineq_save_file+'.npy', allow_pickle=True)
        threshold_lst_explain = np.load(rule_threshold_save_file+'.npy', allow_pickle=True)
        time_explain = np.load(rule_time_save_file+'.npy')





    for j in range(len(feature_lst_explain)):
        feature = feature_lst_explain[j]
        inequality = inequality_lst_explain[j]
        threshold = threshold_lst_explain[j]

        print(j, ['feature=' + str(feature[k]) + ' ' + inequality[k]+ str(round(threshold[k], 3)) for k in range(len(feature))])
        # print(feature, 'explain_ins_fea=', explained_data[j, feature], inequality, threshold)
        # batch_idx = int(explained_idx[j] / config.batch_size) - 1
        # proto_name = f'batchidx_{batch_idx}'
        # protos = np.load(proto_path + '/' + proto_name + '.npy', allow_pickle=True)
        # idx_lst = sum([proto.data_idx_lst for proto in protos], [])
        # idx_array = np.array(idx_lst)
        # normal_values = data_use[idx_array][:, feature]
        # print('normal_min=',[np.min(normal_values[:,k]) for k in range(normal_values.shape[1])])
        # print('normal_max=',[np.max(normal_values[:,k]) for k in range(normal_values.shape[1])])

    # for i in range(len(feature_lst_explain)):
    #     print('rule:', 12, inequality_lst_explain[i], threshold_lst_explain[i])
    #     print([np.min(data_use[:,12]), np.percentile(data_use[:,12], 99), np.max(data_use[:,12])], explained_data[i, 12])
    #     if i == 2:
    #         exit()

    # print('Begin Rule Evaluation')
    # results = Parallel(n_jobs=-1)(
    #     delayed(stream_rule_evaluate)(args,
    #                                   config,
    #                                   explained_data[i],
    #                                   explained_label[i],
    #                                   explained_idx[i],
    #                                   explained_data,
    #                                   explained_label,
    #                                   feature_lst_explain[i],
    #                                   inequality_lst_explain[i],
    #                                   threshold_lst_explain[i],
    #                                   proto_path,
    #                                   data_use,
    #                                   label_use) for i in range(len(explained_data)))

    # roc_lst = []
    # pr_lst = []
    # f1_lst = []
    # attack_precision_lst = []
    # attack_recall_lst = []
    # for i in range(len(results)):
    #     roc_lst.append(results[i][0])
    #     pr_lst.append(results[i][1])
    #     f1_lst.append(results[i][2])
    #     attack_precision_lst.append(results[i][3])
    #     attack_recall_lst.append(results[i][4])
    #
    # roc_mean = np.mean(roc_lst)
    # pr_mean = np.mean(pr_lst)
    # f1_mean = np.mean(f1_lst)
    # attack_precision_mean = np.mean(attack_precision_lst)
    # attack_recall_mean = np.mean(attack_recall_lst)


    # fea_len = 0
    # for i in range(len(feature_lst_explain)):
    #     fea_len += len(set(feature_lst_explain[i]))
    # fea_len = fea_len / len(feature_lst_explain)
    # print(f'This is {args.exp_method}')
    # print('roc_auc_mean_value=', round(roc_mean, 3))
    # print('pr_mean_value=', round(pr_mean, 3))
    # print('f1_mean_value=', round(f1_mean, 3))
    # print('attack_precsion_mean_value=', round(attack_precision_mean, 3))
    # print('attack_recall_mean_value=', round(attack_recall_mean, 3))
    # print('avg_fea_len=', round(fea_len, 3))
    # print('time_explain=', time_explain)

    # save_rule_results(args, config, round(roc_mean, 3), round(pr_mean, 3), round(f1_mean, 3), round(attack_precision_mean, 3), round(attack_recall_mean, 3),
    #                   round(fea_len, 3), len(explained_data), time_explain)


if __name__ == '__main__':
    arg_parser = get_comparative_parser()

    arg_parser.add_argument('--datatype', type=str, help="Dataset type.", default='NADC')
    arg_parser.add_argument('--train_date', type=str, help="train data date", default='1203')
    arg_parser.add_argument('--device', type=str, help="Torch device.", default='cuda:0',
                            choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    arg_parser.add_argument('--exp_method', type=str, help="train data date", default='Enids')
    arg_parser.add_argument('--eps', type=float, help="error rate", default=1e-6)
    arg_parser.add_argument('--explain_flag', type=str, help="", default="feature")
    arg_parser.add_argument('--feat', type=list, help="", default=0.9)
    arg_parser.add_argument('--attack_label', type=int, help="", default=2)
    arg_parser.add_argument('--train_all_data', type=str, help="", default='False')
    arg_parser.add_argument('--attack_upper_num', type=int, help="", default=100)


    # feature
    arg_parser.add_argument('--a_ipdt', type=float, help="", default=1.0)
    arg_parser.add_argument('--epochs', type=int, help="", default=500)
    arg_parser.add_argument('--lr', type=float, help="", default=0.1)
    arg_parser.add_argument('--alpha', type=float, help="", default=0.01)
    arg_parser.add_argument('--t1', type=float, help="", default=0.5)
    arg_parser.add_argument('--t2', type=float, help="", default=0.05)
    arg_parser.add_argument('--store_w', type=str, help="", default='True')


    # rule
    arg_parser.add_argument('--loss_guided', type=str, help="", default='True')
    arg_parser.add_argument('--keypoints', type=str, help="", default='False')
    arg_parser.add_argument('--percentile', type=float, help="", default=10.0)
    arg_parser.add_argument('--tolerance', type=float, help="", default=0.5)







    parsed_args = arg_parser.parse_args()


    main(parsed_args)
