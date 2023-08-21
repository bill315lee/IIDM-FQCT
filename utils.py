import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math
import numpy as np
from collections import Counter
import os
import numba as nb
import socket
import pandas as pd

@nb.njit()
def MatrixEuclideanDistances(a, b):
    sq_a = a ** 2
    sum_sq_a = np.sum(sq_a,axis=1).reshape(-1,1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = np.sum(sq_b,axis=1).reshape(1,-1)  # n->[1, n]
    bt = b.T

    # return np.clip(sum_sq_a+sum_sq_b-2*a.dot(bt), 0, a_max=None)
    return sum_sq_a+sum_sq_b-2*a.dot(bt)


def gen_pseudo_points_in_proto(center, r, k):
    dim = len(center)
    pseudo_points = []

    for _ in range(k):
        rho = np.random.uniform(0, np.pi, dim)
        rho[0] = rho[0] * r / np.pi
        rho[1] = rho[1] * 2
        x = np.zeros(dim)
        for i in range(dim-1):
            x[i] = rho[0] * math.cos(rho[dim-1-i])
            rho[0] = rho[0] * math.sin(rho[dim-1-i])

        x[-1] = rho[0]

        print(euclidian_dist(x, center), r)


        pseudo_points.append(x)

    return np.array(pseudo_points)


def get_feature_names(datatype):
    if datatype == 'IDS2017':
        feature_names = ['A', 'Flow_Duration', 'Total_Fwd_Packet', 'Total_Bwd_packets', 'Total_Length_of_Fwd_Packet', 'Total_Length_of_Bwd_Packet',
                         'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean', 'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max',
                         'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean',
                         'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
                         'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags',
                         'Bwd_URG_Flags', 'Fwd_RST_Flags', 'Bwd_RST_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
                         'Packet_Length_Min', 'Packet_Length_Max', 'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
                         'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWR_Flag_Count', 'ECE_Flag_Count',
                         'Down/Up_Ratio', 'Average_Packet_Size', 'Fwd_Segment_Size_Avg', 'Bwd_Segment_Size_Avg', 'Fwd_Bytes/Bulk_Avg', 'Fwd_Packet/Bulk_Avg',
                         'Fwd_Bulk_Rate_Avg', 'Bwd_Bytes/Bulk_Avg', 'Bwd_Packet/Bulk_Avg', 'Bwd_Bulk_Rate_Avg', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
                         'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes', 'FWD_Init_Win_Bytes', 'Bwd_Init_Win_Bytes', 'Fwd_Act_Data_Pkts', 'Fwd_Seg_Size_Min',
                         'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

    elif datatype == 'NADC':

        feature_names = ["duration", "out_bytes", "in_bytes", "proto",
         "cnt_dst", "cnt_src", "cnt_serv_src", "cnt_serv_dst", "cnt_dst_slow", "cnt_src_slow",
         "cnt_serv_src_slow", "cnt_serv_dst_slow", "cnt_dst_conn", "cnt_src_conn", "cnt_serv_src_conn",
         "cnt_serv_dst_conn"]

    elif datatype == 'shuttle':
        feature_names = [0]*9
    elif datatype == 'NYTimes':
        feature_names = [0]*100
    elif datatype == 'creditcard':
        feature_names = [0]*29
    elif datatype == 'har':
        feature_names = [0]*561

    elif datatype == 'usps':
        feature_names = [0]*256

    elif datatype == 'toy':
        feature_names = ["A0", "A1", "A2", "A3", "A4", "A5"]


    return feature_names


def get_data_path_prefix():

    data_path_prefix = "/home/lb/data/IntrusionData"


    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.connect(("8.8.8.8", 80))
    # ip_addr = s.getsockname()[0]
    # if ip_addr == '10.107.21.241':
    #     data_path_prefix = '/media/sda/bill'
    # elif ip_addr == '10.107.21.242':
    #     data_path_prefix = '/media/sda/bill'
    # elif ip_addr == '10.0.9.66':
    #     data_path_prefix = '/sda/bill'
    # elif ip_addr == '10.0.9.49':
    #     data_path_prefix = '/mnt/IntrusionData'
    # elif ip_addr == '10.107.1.20':
    #     data_path_prefix = '/home/lb/bill'
    # else:
    #     print(ip_addr)
    #     print('Unknown Address')
    #     raise NotImplementedError
    #

    return data_path_prefix

def read_train_test_data_stream(args):

    data_path_prefix = get_data_path_prefix()
    if args.datatype == 'IDS2017':
        prefix_path = os.path.join(data_path_prefix, args.datatype, args.datatype + '_fix', 'remain')
        data_path = f'{prefix_path}/{args.train_date}_drop_data.npy'
        label_path = f'{prefix_path}/{args.train_date}_drop_label.npy'
        four_tag_path = f'{prefix_path}/{args.train_date}_drop_four_tag.npy'

    elif args.datatype == 'NADC':
        file_name = f'{args.train_date}_labelprocess'
        data_path = os.path.join(data_path_prefix, 'NADC', file_name + '.npz')
        label_path = None
        four_tag_path = None

    elif args.datatype == 'toy' and args.train_date in ['1', '2', '3']:
        file_name = f'{args.datatype}{args.train_date}'
        data_path = os.path.join('/home', 'lb', 'project', 'explain_ids', 'toydata', file_name + '.npz')

    elif args.datatype in ['creditcard', 'shuttle', 'har', 'usps', 'NYTimes']:
        prefix_path = '/home/lb/data/IntrusionData/'


    else:
        raise NotImplementedError


    if args.datatype == "IDS2017":

        data_end = np.load(data_path)
        labels = np.load(label_path, allow_pickle=True)
        four_tag = np.load(four_tag_path, allow_pickle=True)

        data_mu = np.mean(data_end, axis=0)
        data_std = np.std(data_end, axis=0)
        data_process = (data_end - data_mu) / (data_std + 1e-12)


    elif args.datatype == "NADC":

        if args.train_date != '1216':
            data_infor = np.load(data_path, allow_pickle=True)
            data_end = data_infor['data']
            labels = data_infor['label']
            four_tag = data_infor['four_tag']
        else:
            data_infor = np.load(data_path, allow_pickle=True)
            data_end = data_infor['data'][1400000:]
            labels = data_infor['label'][1400000:]
            four_tag = data_infor['four_tag']

        data_mu = np.mean(data_end, axis=0)
        data_std = np.std(data_end, axis=0)
        data_process = (data_end - data_mu) / (data_std + 1e-10)




    elif args.datatype == "toy":
        data_label = np.load(data_path)
        data_process = data_label['x']
        labels = data_label['y']


    elif args.datatype in ['creditcard', 'shuttle', 'NYTimes', 'har', 'usps']:
        data_label = pd.read_csv(os.path.join(prefix_path, f'{args.datatype}/data_label_stream.csv'), header=None).values

        data_process = data_label[:, :-1]
        labels = data_label[:, -1]-1
        data_mu = np.mean(data_process, axis=0)
        data_std = np.std(data_process, axis=0)

        data_process = (data_process - data_mu) / (data_std + 1e-10)


    else:
        raise NotImplementedError("not implemented the dataset")

    # anomaly_ratio = round(
    #     len(np.where(labels != 0)[0]) / (len(np.where(labels == 0)[0]) + len(np.where(labels != 0)[0])), 4)
    # print(data_process.shape, Counter(labels), anomaly_ratio)
    # exit()


    attack_idx = np.where(labels != 0)[0]
    attack_data = data_process[attack_idx]
    attack_label = labels[attack_idx]
    attack_dict = {}
    for label_key in set(attack_label):
        attack_dict[label_key] = attack_data[np.where(attack_label == label_key)[0]]
    

    print('data_size=', data_process.shape)


    if args.datatype in ['IDS2017', 'NADC']:
        data_use = data_process[:args.stream_size]
        label_use = labels[:args.stream_size]
    else:
        data_use = data_process
        label_use = labels


    

    return data_use, label_use, data_mu, data_std, attack_dict



def read_train_test_data_sample(args):

    if args.datatype == 'IDS2017':
        prefix_path = os.path.join('/media/sda', 'bill', args.datatype, args.datatype + '_fix', 'remain')
        data_path = f'{prefix_path}/{args.train_date}_drop_data.npy'
        label_path = f'{prefix_path}/{args.train_date}_drop_label.npy'
        four_tag_path = f'{prefix_path}/{args.train_date}_drop_four_tag.npy'

    elif args.datatype == 'NADC':
        file_name = f'{args.train_date}_labelprocess'
        data_path = os.path.join('/media/sda', 'bill', 'NADC', file_name + '.npz')
        label_path = None
        four_tag_path = None

    elif args.datatype == 'toy' and args.train_date in ['1', '2']:
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

        # min_value = np.min(data_end, axis=0)
        # max_value = np.max(data_end, axis=0)
        # data_process = (data_end - min_value) / (max_value - min_value + 1e-8)


        data_mu = 0
        data_std = 0

    elif args.datatype == "NADC":
        data_infor = np.load(data_path, allow_pickle=True)
        data_end = data_infor['data']
        print(data_end.shape)
        labels = data_infor['label']
        four_tag = data_infor['four_tag']
        # timestamp = data_infor['time_stamp']

        data_mu = np.mean(data_end, axis=0)
        data_std = np.std(data_end, axis=0)
        data_process = (data_end - data_mu) / (data_std + 1e-10)
        # min_value = np.min(data_end, axis=0)
        # max_value = np.max(data_end, axis=0)
        # data_process = (data_end - min_value) / (max_value - min_value + 1e-8)

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

    attack_data = data_process[np.where(labels != 0)[0]]
    attack_label = labels[np.where(labels != 0)[0]]
    attack_dict = {}
    for label_key in set(attack_label):
        attack_dict[label_key] = attack_data[np.where(attack_label == label_key)[0]]


    if args.datatype == "toy":
        attack_idx_array = np.where(labels != 0)[0]
    else:
        attack_idx = np.where(labels != 0)[0]

        attack_label_set = set(labels[attack_idx])
        attack_idx_lst = []
        for label in attack_label_set:
            attack_idx_lst.extend(list(np.where(labels == label)[0][:100]))

        attack_idx_array = np.array(attack_idx_lst)

    data_train = data_process[norma_idx]
    label_train = np.array(labels[norma_idx], dtype=int)

    data_test = data_process[attack_idx_array]
    label_test = np.array(labels[attack_idx_array], dtype=int)




    print('train shape=', data_train.shape, Counter(label_train))
    print('test shape=', data_test.shape, Counter(label_test))


    return data_train, label_train, data_test, label_test, data_mu, data_std, attack_dict

def num_to_subspace(n):
    n = list(str(n))[::-1]
    n = [int(a) for a in n]

    subspace = []
    for i in range(len(n)):
        if n[i] == 1:
            subspace.append(i)

    return subspace


def subspace_decision(fea):
    score_new_lst = []
    for i in range(len(fea)):
        remain_idx = np.array(list(set(np.arange(len(fea))) - set([i])))
        score_abs = np.abs(fea[i] - fea[remain_idx])
        min_score = np.min(score_abs)
        max_score = np.max(score_abs)
        score_new = min_score / max_score
        score_new_lst.append(score_new)

    score_new_array = np.array(score_new_lst)

    print(score_new_array)
    threshold = np.sqrt(2/ len(score_new_array)) * np.sum(score_new_array)
    print(threshold)
    fea_sort_idx = np.argsort(-score_new_array)
    score = 0
    idx_lst = []
    for idx in fea_sort_idx:
        score += score_new_array[idx]
        idx_lst.append(idx)
        if score <= threshold:
            pass
        else:
            break

    return idx_lst


def SD_diff(score, nbr_num):
    score_new_lst = []
    for i in range(len(score)):
        remain_idx = np.array(list(set(np.arange(len(score))) - set([i])))
        score_abs = np.abs(score[i] - score[remain_idx])
        min_score = np.mean(np.sort(score_abs)[:nbr_num])
        max_score = np.mean(np.sort(score_abs)[-nbr_num:])
        score_new = min_score / max_score
        score_new_lst.append(score_new)

    score_new_array = np.array(score_new_lst)
    # 从大到小排序
    score_sort = np.sort(score_new_array)[::-1]

    threshold_lst = []
    for k in range(1, len(score_sort)):
        thre = np.abs(np.std(score_sort[:k]) - np.std(score_sort[k:]))
        threshold_lst.append(thre)

    threshold_idx = np.argmin(threshold_lst)

    idx = np.where(score_new_array >= score_sort[threshold_idx])[0]


    return idx





def gen_pseudo_points_in_proto(center, r, k):
    dim = len(center)
    rho = np.random.rand(k, 1) * r
    theta = np.random.rand(k, dim - 1)
    cos_values = np.cos(theta)  # (n, dim-1)
    cos_values = np.concatenate([cos_values, np.ones(shape=(k, 1))], axis=1)  # (n, dim)

    sin_values = np.sin(theta)  # (n, dim-1)
    sin_values = np.concatenate([np.ones(shape=(k, 1)), sin_values], axis=1)  # (n, dim)
    cum_sin_values = np.cumprod(sin_values, axis=1)  # (n, dim)
    offsets = cum_sin_values * cos_values * rho
    points = offsets + center[None]

    return points




def get_mahalanobis(invD, a, b):

    X = (a - b).T
    return np.sqrt(np.dot(np.dot(X.T, invD), X))

def str_to_bool(str):
    return True if str == 'True' else False

def plot_figure_effects_SA(data_train, attack_train, data_test, f_idx, path):
    """
    :param Mat: 二维点坐标矩阵
    :param Label: 点的类别标签
    :return:
    """
    plt.clf()
    plt.scatter(data_train[:, 0], data_train[:, 1], s=2, c='b', marker='o')
    plt.scatter(attack_train[:, 0], attack_train[:, 1], s=3, c='r', marker=',')

    color = ['g',  'c', 'm', 'y', 'k', 'darkred', 'chocolate', 'yellow','pink']
    marker = ['.', 'o','v','1','2','+','x', 'D', '|']


    plt.scatter(data_test[0, 0], data_test[0, 1], s=10, c=color[0], marker=marker[0])
    # plt.scatter(counterfactual[i, 0], counterfactual[i, 1], s=10, c=color[i], marker=marker[i])


    plt.axis()
    plt.xlabel(str(f_idx[0]))
    plt.ylabel(str(f_idx[1]))
    plt.savefig(path, bbox_inches='tight', dpi=1024)

def predictor_train(data_all_df, label_all_df, sample_weight=None):

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(data_all_df, label_all_df, sample_weight=sample_weight)
    # pred = clf.predict(data_all_df)
    # print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(label_all_df, pred)))

    return clf


# def nn_model(dim):
#
#     x_in = Input(shape=(dim,))
#     x = Dense(40, activation='relu')(x_in)
#     x = Dense(40, activation='relu')(x)
#     x_out = Dense(2, activation='softmax')(x)
#     nn = Model(inputs=x_in, outputs=x_out)
#     nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     return nn

def euclidian_dist(a, b):

    if a.ndim == 1:
        a = a.reshape(1,-1)

    if b.ndim == 1:
        b = b.reshape(1, -1)

    return np.sqrt(np.clip(np.sum((a - b) ** 2, axis=1),0, a_max=None) + 1e-8)


def saving_explanation_results(args, path, batch_id, output, end_tag=False):

    with open(path, 'a+') as f:
        if end_tag == False:
            f.write("\n")
            f.write("{}-{}-{}".format(args.datatype, args.train_date, str(batch_id)) + "\n")
            f.write("base_classifier={}".format(args.active_method) + "\n")
            f.write("result=" + str(output) + "\n")
        else:
            f.write("-----------------------------------------" + "\n")



def rule_process( feature_lst_explain, inequality_lst_explain, threshold_lst_explain):

    feature_lst_all_data = []
    inequality_lst_all_data = []
    threshold_lst_all_data = []
    for i in range(len(feature_lst_explain)):

        feature_lst = feature_lst_explain[i]
        inequality_lst = inequality_lst_explain[i]
        threshold_lst = threshold_lst_explain[i]
        min_dict = {}
        max_dict = {}
        for j in range(len(feature_lst)):
            feature = feature_lst[j]
            inequality = inequality_lst[j]
            threshold = threshold_lst[j]

            if inequality in ['<', '<=']:
                if feature in max_dict:
                    now = max_dict[feature]
                    if threshold <= now:
                        max_dict[feature] = threshold
                else:
                    max_dict[feature] = threshold

            elif inequality in ['>=','>']:
                if feature in min_dict:
                    now = min_dict[feature]
                    if threshold >= now:
                        min_dict[feature] = threshold
                else:
                    min_dict[feature] = threshold
            else:
                print('Emerging ==')
                raise NotImplementedError

        feature_lst = []
        inequality_lst = []
        threshold_lst = []
        for fea in min_dict:
            feature_lst.append(fea)
            inequality_lst.append('>=')
            threshold_lst.append(min_dict[fea])

        for fea in max_dict:
            feature_lst.append(fea)
            inequality_lst.append('<')
            threshold_lst.append(max_dict[fea])


        feature_lst_all_data.append(feature_lst)
        inequality_lst_all_data.append(inequality_lst)
        threshold_lst_all_data.append(threshold_lst)

    return feature_lst_all_data, inequality_lst_all_data, threshold_lst_all_data

