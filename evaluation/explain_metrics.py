from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score,confusion_matrix
import numpy as np
from itertools import combinations
import copy
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from utils import str_to_bool
import torch.nn as nn
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from evaluation.evaluate_tools import roc_auc, pr, best_f1, ptopk
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from pyod.models.ecod import ECOD
from collections import Counter
from utils import saving_explanation_results, euclidian_dist
import random
import sklearn
rng = np.random.RandomState(42)




def evaluation_feature_important(args, data_train, label_train, data_test, label_test, stream_file_lst, file_prefix, feature_names):
    normal_data_global = []
    normal_idx = np.where(label_train == 0)[0]
    normal_data_global.extend(list(data_train[normal_idx]))

    print("This is", args.exp_method, "!!!!!")
    if args.exp_method == 'para_proto':
        exp_parameter = f'{args.ipdt}'
    elif args.exp_method == 'Deepaid':
        exp_parameter = f'{str(args.ae_t)}'
    else:
        exp_parameter = ""

    if str_to_bool(args.only_update_label):
        update_tag = "_only_update_label"
    else:
        update_tag = "only_update_label"

    save_exp_feature_weight_path = os.path.join('.', 'feature_weight_stream'+update_tag,
                                                f'{args.active_method}_train_{args.train_size}_batch_{args.batch_size}_qr_{args.query_ratio}_normthre_{args.normal_threshold}/{args.exp_method}')

    if str_to_bool(args.only_test_label):
        tag = "test_label"
    else:
        tag = "test_all"
    output_file_name = f'./results/{args.datatype}_{args.train_date}_{args.exp_method}_{tag}'


    attack_data_history = []
    attack_label_history = []

    label_iforest_dict = {}
    label_hbos_dict = {}
    label_ecod_dict = {}

    label_dist_dict = {}
    label_sparse_dict = {}


    for j in range(1, len(stream_file_lst)):
        print(f'-------- {j}-th batch --------')
        if j * args.batch_size > len(data_test):
            X_batch = data_test[(j-1)*args.batch_size:len(data_test)]
            y_batch = label_test[(j-1)*args.batch_size:len(data_test)]
        else:
            X_batch = data_test[(j-1)*args.batch_size:j*args.batch_size]
            y_batch = label_test[(j-1)*args.batch_size:j*args.batch_size]

        if str_to_bool(args.only_test_label):
            stored_idx = np.load(os.path.join(file_prefix, str(j) + '.npz'))
            normal_idx_batch = list(stored_idx['labeled_normal_idx'])

            normal_data_batch = data_test[normal_idx_batch]
            attack_idx_batch = np.where(y_batch!=0)[0]
            attack_data_batch = X_batch[attack_idx_batch]
            attack_label_batch = y_batch[attack_idx_batch]
        else:
            normal_idx_batch = np.where(y_batch == 0)[0]
            normal_data_batch = X_batch[normal_idx_batch]
            attack_idx_batch = np.where(y_batch != 0)[0]
            attack_data_batch = X_batch[attack_idx_batch]
            attack_label_batch = y_batch[attack_idx_batch]


        attack_idx = np.where(y_batch != 0)[0]
        print(f'update_tag={update_tag}')

        if len(attack_idx) == 0:
            pass
        else:
            fw_t_1 = np.load(os.path.join(save_exp_feature_weight_path, f'{j}_{args.datatype}_{args.train_date}_{exp_parameter}_feature_weight_0.npz'), allow_pickle=True)
            # fw_t_2 = np.load(os.path.join(save_exp_feature_weight_path, f'{j}_{args.datatype}_{args.train_date}_{exp_parameter}_feature_weight_1.npz'), allow_pickle=True)
            # fw_t_3 = np.load(os.path.join(save_exp_feature_weight_path, f'{j}_{args.datatype}_{args.train_date}_{exp_parameter}_feature_weight_2.npz'), allow_pickle=True)

            fea_weight_dict1 = fw_t_1['feature_weight'].item()
            time1 = fw_t_1['time']
            # fea_weight_dict2 = fw_t_2['feature_weight'].item()
            # time2 = fw_t_2['time']
            # fea_weight_dict3 = fw_t_3['feature_weight'].item()
            # time3 = fw_t_3['time']
            # time_avg = (time1+time2+time3)/3

            attack_idx_dict = fw_t_1['attack_idx'].item()


            normal_data_global_array = np.array(normal_data_global)
            # attack_data_history_array = np.array(attack_data_history)
            # attack_label_history_array = np.array(attack_label_history)


            for label_key in fea_weight_dict1:
                fea_weight_class1 = np.array(fea_weight_dict1[label_key])
                # fea_weight_class2 = np.array(fea_weight_dict2[label_key])
                # fea_weight_class3 = np.array(fea_weight_dict3[label_key])
                explain_attack_idx_class = attack_idx_dict[label_key]
                # f_weights = [fea_weight_class1, fea_weight_class2, fea_weight_class3]
                # f_weight_average = (fea_weight_class1 + fea_weight_class2 + fea_weight_class3) / 3
                f_weights = fea_weight_class1
                f_weight_average = fea_weight_class1
                # 评测的是这个batch中这个label_key的攻击


                # 对比数据中包含攻击
                # other_attacks_data = attack_data_history_array[np.where(attack_label_history_array!=label_key)[0]]
                # other_attacks_label = attack_label_history_array[np.where(attack_label_history_array!=label_key)[0]]
                # if len(other_attacks_data) != 0:
                #     dataall = np.vstack((normal_data_global_array, other_attacks_data))
                #     labelall = np.array(list(np.zeros(len(normal_data_global_array)))+list(other_attacks_label))
                # else:
                #     dataall = normal_data_global_array
                #     labelall = np.zeros(len(normal_data_global_array))

                # 现阶段就是只有正常数据
                dataall = normal_data_global_array
                labelall = np.zeros(len(normal_data_global_array))

                explain_data = X_batch[explain_attack_idx_class]
                explain_label = y_batch[explain_attack_idx_class]
                assert (0 not in explain_label) and (len(set(explain_label)) == 1)

                explain_data_iforest_array, explain_data_hbos_array,explain_data_ecod_array = \
                    effectivness_global_metric(dataall, labelall, explain_data, explain_label, f_weight_average, args.k_lst, feature_names)

                if label_key in label_iforest_dict:
                    label_iforest_dict[label_key] = np.hstack((label_iforest_dict[label_key], explain_data_iforest_array))
                    label_hbos_dict[label_key] = np.hstack((label_hbos_dict[label_key], explain_data_hbos_array))
                    label_ecod_dict[label_key] = np.hstack((label_ecod_dict[label_key], explain_data_ecod_array))
                else:
                    label_iforest_dict[label_key] = explain_data_iforest_array
                    label_hbos_dict[label_key] = explain_data_hbos_array
                    label_ecod_dict[label_key] = explain_data_ecod_array



                explain_data_dist_array = dist_predicted_subspace(dataall, labelall, explain_data, explain_label, f_weight_average, args.k_lst, feature_names)
                if label_key in label_dist_dict:
                    label_dist_dict[label_key] = np.hstack((label_dist_dict[label_key], explain_data_dist_array))
                else:
                    label_dist_dict[label_key] = explain_data_dist_array


                sparse_array = sparsity_metric(f_weight_average)
                if label_key in label_sparse_dict:
                    label_sparse_dict[label_key].extend(list(sparse_array))
                else:
                    label_sparse_dict[label_key] = list(sparse_array)


        normal_data_global.extend(list(normal_data_batch))
        attack_data_history.extend(list(attack_data_batch))
        attack_label_history.extend(list(attack_label_batch))



        if args.train_date == '1203':
            if j == 20:
                results_print(label_sparse_dict, label_dist_dict, label_iforest_dict, label_hbos_dict, label_ecod_dict, j)
                break
        elif args.train_date in ['1210', '1216']:
            if j == 40:
                results_print(label_sparse_dict, label_dist_dict, label_iforest_dict, label_hbos_dict, label_ecod_dict, j)
                break





def results_print(label_sparse_dict, label_dist_dict, label_iforest_dict, label_hbos_dict, label_ecod_dict, batch_idx):
    print(f'------------{batch_idx}-----------')
    print('Sparse Average')
    for label_key in label_sparse_dict:
        print('attack=', label_key, np.mean(label_sparse_dict[label_key]))
    print('Dist Average')
    for label_key in label_dist_dict:
        print('attack=', label_key, [np.mean(row) for row in label_dist_dict[label_key]])
    print('AUCROC Average-iForest')
    for label_key in label_iforest_dict:
        print('attack=', label_key, [np.mean(row) for row in label_iforest_dict[label_key]])
    print('AUCROC Average-HBOS')
    for label_key in label_hbos_dict:
        print('attack=', label_key, [np.mean(row) for row in label_hbos_dict[label_key]])
    print('AUCROC Average-ECOD')
    for label_key in label_ecod_dict:
        print('attack=', label_key, [np.mean(row) for row in label_ecod_dict[label_key]])

def evaluate_detection(truth, preds, average='macro'):
    acc = accuracy_score(truth, preds)
    precision = precision_score(y_true=truth, y_pred=preds, average=average)
    recall = recall_score(y_true=truth, y_pred=preds, average=average)
    f1 = f1_score(y_true=truth, y_pred=preds, average=average)
    cm = confusion_matrix(y_true=truth, y_pred=preds, labels=sorted(list(np.unique(truth))))


    return acc, precision, recall, f1


def explanation_similarities(f_weights_1, f_weights_2, k):

    f_topk_1 = np.argsort(-f_weights_1)[:k]
    f_topk_2 = np.argsort(-f_weights_2)[:k]
    sim_value = 2 * len(np.intersect1d(f_topk_1,f_topk_2)) / (len(f_topk_1)+len(f_topk_2))

    return sim_value



def stability_metric(f_weights_run, k_lst):

    f_weights_lst = [[] for _ in range(len(f_weights_run[0]))]

    for i in range(len(f_weights_run)):
        for j in range(len(f_weights_run[i])):
            f_weights_lst[j].append(f_weights_run[i][j])

    stability_value_lst = []
    for k in k_lst:
        stability_value = 0
        for i in range(len(f_weights_lst)):
            f_weights_method = f_weights_lst[i]
            tuple_lst = list(combinations(f_weights_method, 2))
            stability_value_sample = 0
            for tuple in tuple_lst:
                stability_value_sample += explanation_similarities(tuple[0], tuple[1], k)

            stability_value += (2 * stability_value_sample / (len(f_weights_method) * (len(f_weights_method) - 1)))
        stability_value_lst.append(stability_value)


    return np.mean(stability_value_lst)




def robustness_metric(label, f_weights, k):

    assert len(label) == len(f_weights)

    label_set = set(label)
    same_label_idx_dict = {}
    diff_label_idx_dict = {}
    for l in label_set:
        idx = np.where(label == l)[0]
        same_label_idx_dict[l] = idx
        diff_label_idx_dict[l] = np.array(list(set(np.arange(len(label))) - set(idx)))

    robustness_value = 0
    for i in range(len(f_weights)):
        # only focus the effects from attack instance on the robustness
        if label[i]!=0:
            same_label_f = f_weights[same_label_idx_dict[label[i]]]
            diff_label_f = f_weights[diff_label_idx_dict[label[i]]]
            # print('f_id=',i, 's_f=', same_label_f, 'd_f=',diff_label_f)
            S_avg = 0
            d_avg = 0
            for j in range(len(same_label_f)):
                S_avg += explanation_similarities(f_weights[i], same_label_f[j], k)

            for jj in range(len(diff_label_f)):
                d_avg += explanation_similarities(f_weights[i], diff_label_f[jj], k)

            robustness_value += (S_avg / len(same_label_f) - d_avg / len(diff_label_f))



    return robustness_value
# 这个值得商榷
# 这个是典型的边界评价标准的方法



def dist_predicted_subspace(data_train, label_train, explain_data, explain_label, feature_weights, k_lst, feature_names):

    N = len(explain_data)
    dist_lst_k = []
    for k in k_lst:
        explain_data_new = pickle.loads(pickle.dumps(explain_data))
        dist_lst = []
        for i in range(N):
            few_sorts = np.argsort(-feature_weights[i])[:k]
            dist = euclidian_dist(explain_data_new[i, few_sorts], data_train[:,few_sorts])
            dist_lst.append(np.mean(dist))
        dist_lst_k.append(dist_lst)


    # 返回的是 K*N 的矩阵，k是[1,3,5]分别代表的预测特征，N代表的是有多少个待解释数据
    return np.array(dist_lst_k)


def effectivness_supervised_IDS(data_train, label_train, explain_data, explain_label, feature_weights, k_lst, feature_names):

    rf = RandomForestClassifier(n_estimators=50, random_state=rng,n_jobs=-1)
    data_all = np.vstack((data_train, explain_data))
    label_all = np.array(list(label_train)+list(explain_label))
    rf.fit(data_all, label_all)

    pred = rf.predict(data_all)
    print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(label_all, pred)))

    data_min = np.min(data_train, axis=0)
    data_max = np.max(data_train, axis=0)

    wrong_num_lst_k = []
    for k in k_lst:
        wrong_num_lst = []
        for _ in range(10):
            explain_data_new = pickle.loads(pickle.dumps(explain_data))
            for i in range(len(explain_data_new)):
                permute_idx = np.argsort(-feature_weights[i])[:k]

                for p_idx in permute_idx:
                    explain_data_new[i, p_idx] = random.uniform(data_min[p_idx], data_max[p_idx])

            predicts = rf.predict(explain_data_new)

            wrong_num = len(np.where(predicts != explain_label)[0])

            wrong_num_lst.append(wrong_num)

        wrong_num_lst_k.append(np.mean(wrong_num_lst))

    return np.mean(wrong_num_lst_k)


def effectivness_supervised_toy(data_train, label_train, explain_data, explain_label, feature_weights, train_date):
    rf = RandomForestClassifier(n_estimators=50)
    data_all = np.vstack((data_train, explain_data))
    label_all = np.array(list(label_train)+list(explain_label))
    rf.fit(data_all, label_all)
    data_min = np.min(data_train, axis=0)
    data_max = np.max(data_train, axis=0)

    wrong_num_lst = []
    for _ in range(10):
        explain_data_new = pickle.loads(pickle.dumps(explain_data))
        if train_date == '1':
            for i in range(len(explain_data_new)):
                few_sorts = np.argsort(-feature_weights[i])
                if i < 30 or (i< 120 and i >= 90):
                    permute_idx = few_sorts[0]
                    explain_data_new[i,permute_idx] = random.uniform(data_min[permute_idx], data_max[permute_idx])
                elif (i < 60 and i >= 30) or (i < 150 and i >= 120):
                    permute_idx = few_sorts[0]
                    explain_data_new[i,permute_idx] = random.uniform(data_min[permute_idx], data_max[permute_idx])
                elif (i < 90 and i >= 60) or (i< 180 and i >= 150):
                    permute_idx = few_sorts[:2]
                    for p_idx in permute_idx:
                        explain_data_new[i,p_idx] = random.uniform(data_min[p_idx], data_max[p_idx])
                else:
                    exit()
        elif train_date == '2':
            for i in range(len(explain_data_new)):
                few_sorts = np.argsort(-feature_weights[i])
                if i < 60:
                    permute_idx = few_sorts[0]
                    explain_data_new[i, permute_idx] = random.uniform(data_min[permute_idx], data_max[permute_idx])
                elif (i < 120 and i >= 60):
                    permute_idx = few_sorts[:2]
                    for p_idx in permute_idx:
                        explain_data_new[i,p_idx] = random.uniform(data_min[p_idx], data_max[p_idx])
                else:
                    exit()



        predicts = rf.predict(explain_data_new)

        wrong_num = len(np.where(predicts != explain_label)[0])
        wrong_num_lst.append(wrong_num)


    return np.mean(wrong_num_lst) / len(explain_data)




def effectivness_global_metric(data_train, label_train, data_explain, label_explain, f_weights, k_lst, feature_names):



    iforest_lst_k_roc = []
    hbos_lst_k_roc = []
    ecod_lst_k_roc = []


    for k in tqdm(k_lst):
        hash_idx_dict = {}
        important_feature_lst = []
        for i in range(len(data_explain)):
            important_feature_idx = np.argsort(-f_weights[i])[:k]
            # print("k=", k, "important_features=", np.array(feature_names)[important_feature_idx])
            important_feature_idx_hash = hash(str(set(important_feature_idx)))
            if important_feature_idx_hash in hash_idx_dict:
                hash_idx_dict[important_feature_idx_hash].append(i)
            else:
                hash_idx_dict[important_feature_idx_hash] = [i]
                important_feature_lst.append(set(important_feature_idx))

        roc_score_iforest_lst = []
        roc_score_hbos_lst = []
        roc_score_ecod_lst = []



        label_all = np.array(list(np.zeros(len(data_train))) + [1])

        for i in range(len(important_feature_lst)):
            important_feature_k = list(important_feature_lst[i])
            explain_idx = hash_idx_dict[hash(str(important_feature_lst[i]))]
            data_train_new = data_train[:, important_feature_k]
            data_explain_new = data_explain[explain_idx][:, important_feature_k]

            iforest = IsolationForest(n_estimators=30, max_samples=256, random_state=rng)
            hbos = HBOS()
            if k == 1:
                ecod = ECOD()
            else:
                ecod = ECOD(n_jobs=-1)

            iforest.fit(data_train_new)
            hbos.fit(data_train_new)
            ecod.fit(data_train_new)

            iforest_normal_score = -iforest.score_samples(data_train_new)
            hbos_normal_score = hbos.decision_function(data_train_new)
            ecod_normal_score = ecod.decision_function(data_train_new)

            iforest_explain_score = -iforest.score_samples(data_explain_new)
            hbos_explain_score = hbos.decision_function(data_explain_new)
            ecod_explain_score = ecod.decision_function(data_explain_new)

            for j in range(len(explain_idx)):

                roc_score_iforest_lst.append(roc_auc(label_all, list(iforest_normal_score)+[iforest_explain_score[j]]))
                roc_score_hbos_lst.append(roc_auc(label_all, list(hbos_normal_score)+[hbos_explain_score[j]]))
                roc_score_ecod_lst.append(roc_auc(label_all, list(ecod_normal_score)+[ecod_explain_score[j]]))

        assert len(roc_score_iforest_lst) == len(data_explain)

        iforest_lst_k_roc.append(roc_score_iforest_lst)
        hbos_lst_k_roc.append(roc_score_hbos_lst)
        ecod_lst_k_roc.append(roc_score_ecod_lst)


    return np.array(iforest_lst_k_roc), np.array(hbos_lst_k_roc), np.array(ecod_lst_k_roc)



def sparsity_metric(f_weights):
    width = 1 / f_weights.shape[1]
    sparse_value_all_sample = []
    for i in range(len(f_weights)):
        important_idx = np.argsort(-f_weights[i])
        f_weights_process = pickle.loads(pickle.dumps(f_weights[i]))
        f_weights_process[np.where(f_weights_process < 0)[0]] = 0

        feature_ratio = f_weights_process / np.sum(f_weights_process)
        sparse_value_lst = []

        for j in range(len(important_idx)):
            sparse_value_lst.append(np.sum(feature_ratio[important_idx[:j]]))
        sparse_value_all_sample.append(sparse_value_lst)

    sparse_value_all_sample_array = np.array(sparse_value_all_sample)

    sparse_array = np.sum(sparse_value_all_sample_array, axis=1) * width

    return sparse_array


def plot_curve_figure(dim, y_value_lst, tag):
    plt.figure()
    x = [i for i in range(dim)]
    plt.plot(x, y_value_lst, 'ro--', label=tag)
    # plt.xticks(x, feature_desc, rotation='80')
    plt.xticks()
    plt.ylabel(tag+'_value')
    plt.xlabel('num_dim')
    plt.legend()
    plt.title(tag+'_metric_f1')
    # plt.savefig(evalfig_file + '_' + tag + '.png', dpi=120)



if __name__ == '__main__':
    # f_weights = np.array([[1,1,1,1,100],[1,1,2,2,100]])
    # value = sparsity_metric(f_weights, './model_' + 'aton' + '/figure'+  '/'+'aton'+'_eval_'+
    #                         'ae'+'_'+ 'IDS2017' + '_' + '2' + '_' + 'new' + '_' +'new')
    # print(value)
    pass

