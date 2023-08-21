import numpy as np
from evaluation.evaluate_tools import roc_auc, pr
from utils import MatrixEuclideanDistances, euclidian_dist, str_to_bool
from tqdm import tqdm
import numba as nb
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)



def toy_rule_evaluate(data_train, data_test, label_test, feature_lst_explain, inequality_lst_explain,
                      threshold_lst_explain):
    roc_all_data = []
    pr_all_data = []


    prec_all_data = []
    recall_all_data = []
    f1_all_data = []
    attack_recall_all_data = []
    attack_precision_all_data = []

    attack_dict = {}

    label_lst = []
    pred_lst = []

    for i in tqdm(range(len(data_test))):

        explained_label = label_test[i]

        # if explained_label in attack_dict:
        #     attack_dict[explained_label].append(data_test[i])
        # else:
        #     attack_dict[explained_label] = [data_test[i]]
        # attacks = np.array(attack_dict[explained_label])

        attacks = data_test[np.where(label_test == explained_label)[0]]


        data = np.vstack((data_train, attacks))
        label = np.array(list(np.zeros(len(data_train))) + list(np.ones(len(attacks))))

        feature_lst = feature_lst_explain[i]
        inequality_lst = inequality_lst_explain[i]
        threshold_lst = threshold_lst_explain[i]

        pred_array = np.zeros(len(data))

        for j in range(len(data)):
            # 所有条件都满足，就是异常，有一个不满足就是正常
            tag = True
            for k in range(len(feature_lst)):
                feature = feature_lst[k]
                inequality = inequality_lst[k]
                threshold = threshold_lst[k]
                if inequality == '<':
                    if data[j][feature] < threshold:
                        pass
                    else:
                        tag = False
                        break

                elif inequality == '>':
                    if data[j][feature] > threshold:
                        pass
                    else:
                        tag = False
                        break

                elif inequality == '>=':
                    if data[j][feature] >= threshold:
                        pass
                    else:
                        tag = False
                        break

                elif inequality == '<=':
                    if data[j][feature] <= threshold:
                        pass
                    else:
                        tag = False
                        break

            if tag == True:
                pred_array[j] = 1
            else:
                pred_array[j] = 0

        roc_value = roc_auc(label, pred_array)
        pr_value = pr(label, pred_array)
        prec_value = precision_score(label, pred_array, average='macro')
        recall_value = recall_score(label, pred_array, average='macro')

        f1_value = f1_score(label, pred_array, average='macro')
        cm = confusion_matrix(y_true=label, y_pred=pred_array, labels=sorted(list(np.unique(label))))
        attack_precision_value = cm[1, 1] / (np.sum(cm[:, 1]) + 1e-6)
        attack_recall_value = cm[1, 1] / (np.sum(cm[1]) + 1e-6)

        label_lst.append(label)
        pred_lst.append(pred_array)

        roc_all_data.append(roc_value)
        pr_all_data.append(pr_value)


        prec_all_data.append(prec_value)
        recall_all_data.append(recall_value)
        f1_all_data.append(f1_value)
        attack_recall_all_data.append(attack_recall_value)
        attack_precision_all_data.append(attack_precision_value)


    i_max = np.argmax(np.array(f1_all_data))
    i_min = np.argmin(np.array(f1_all_data))

    label_max = label_lst[i_max]
    label_min = label_lst[i_min]

    pred_max = pred_lst[i_max]
    pred_min = pred_lst[i_min]




    np.save('../../label_max.npy', label_max)
    np.save('../../label_min.npy', label_min)
    np.save('../../pred_max.npy', pred_max)
    np.save('../../pred_min.npy', pred_min)
    exit()



    return np.mean(roc_all_data), np.mean(pr_all_data), np.mean(prec_all_data), np.mean(recall_all_data), np.mean(f1_all_data), np.mean(attack_precision_all_data), np.mean(attack_recall_all_data)


def feature_evaluate_toy(args, f_imp_array, data_test, label_test, center_array, r_array, data_metric_array):
    for i in range(len(f_imp_array)):
        print(i, 'label=', label_test[i], np.argsort(-f_imp_array[i]))

    incircle_lst = []
    for i in range(len(f_imp_array)):
        explain_instance = data_test[i]
        feature_importance = f_imp_array[i]
        if i < 20:
            imp_idx = np.argsort(-feature_importance)[:1]
        else:
            imp_idx = np.argsort(-feature_importance)[:2]

        replace = np.zeros(len(feature_importance))
        replace[imp_idx] = 1
        remain = 1 - replace
        data_train_new = replace * data_metric_array + remain * explain_instance
        dist_matrix = MatrixEuclideanDistances(data_train_new, center_array)
        inside_num = 0
        for j in range(len(data_train_new)):
            cover_idx = np.where(dist_matrix[j] <= r_array ** 2 + args.eps)[0]
            if len(cover_idx) != 0:
                inside_num += 1
        incircle_lst.append(inside_num / len(data_train_new))

    incircle_lst_mean = np.mean(incircle_lst)

    ground_truth = []
    predict_subspace = []
    pre_lst = []
    jaccard_lst = []
    for i in range(len(f_imp_array)):
        feature_importance = np.array(f_imp_array[i])
        if i >= 0 and i < 10:
            ground_truth = [0]
            predict_subspace = [np.argsort(-feature_importance)[0]]
        elif i >= 10 and i < 20:
            ground_truth = [1]
            predict_subspace = [np.argsort(-feature_importance)[0]]
        elif i >= 20:
            ground_truth = [0, 1]
            predict_subspace = np.argsort(-feature_importance)[:2]
        else:
            pass

        pre = len(np.intersect1d(ground_truth, predict_subspace)) / len(predict_subspace)

        jaccard = len(np.intersect1d(ground_truth, predict_subspace)) / len(
            list(set(ground_truth).union(set(predict_subspace))))
        pre_lst.append(pre)
        jaccard_lst.append(jaccard)
    pre_mean = np.mean(pre_lst)
    jaccard_mean = np.mean(jaccard_lst)

    return incircle_lst_mean, pre_mean, jaccard_mean



def stream_rule_evaluate(args,
                         config,
                         explain_instance,
                         explained_instance_label,
                         explained_instance_idx,
                         explained_data,
                         explained_data_label,
                         feature_lst,
                         inequality_lst,
                         threshold_lst,
                         proto_path,
                         data_use,
                         label_use,
                        ):



    batch_idx = int(explained_instance_idx / config.batch_size) - 1

    proto_name = f'batchidx_{batch_idx}.npy'
    normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)

    # idx_lst = sum([proto.data_idx_lst for proto in normal_protos], [])
    idx_lst = []
    for proto in normal_protos:
        random_idx_lst = list(
            np.random.choice(proto.data_idx_lst, int(len(proto.data_idx_lst)/10), replace=False))
        idx_lst.extend(random_idx_lst)


    idx_array = np.array(idx_lst)
    center_array = np.array([proto.center for proto in normal_protos])
    r_array = np.array([proto.r for proto in normal_protos])

    attacks = []
    explained_data_same_label = explained_data[np.where(explained_data_label == explained_instance_label)[0]]

    dist_matrix = MatrixEuclideanDistances(explained_data_same_label, center_array)
    for j in range(len(dist_matrix)):
        cover_idx = np.where(dist_matrix[j] <= r_array ** 2 + args.eps)[0]
        if len(cover_idx) == 0:
            attacks.append(explained_data_same_label[j])
    attacks = np.array(attacks)

    pass_data = data_use[idx_array]
    pass_label = label_use[idx_array]
    normal_idx = np.where(pass_label == 0)[0]

    data_metric_array = pass_data[normal_idx]

    data = np.vstack((data_metric_array, attacks))
    label = np.array(list(np.zeros(len(data_metric_array))) + list(np.ones(len(attacks))))

    pred_array = rule_compare_final(data, np.array(feature_lst), np.array(inequality_lst), np.array(threshold_lst))



    roc_value = roc_auc(label, pred_array)
    pr_value = pr(label, pred_array)
    f1_value = f1_score(label, pred_array, average='macro')
    cm = confusion_matrix(y_true=label, y_pred=pred_array, labels=sorted(list(np.unique(label))))
    attack_precision_value = cm[1, 1] / (np.sum(cm[:,1]) + args.eps)

    attack_recall_value = cm[1, 1] / (np.sum(cm[1]) + args.eps)

    return roc_value, pr_value, f1_value, attack_precision_value, attack_recall_value


@nb.njit()
def rule_compare_final(data, feature_lst, inequality_lst, threshold_lst):
    pred_array = np.zeros(len(data))
    for j in range(len(data)):
        # 所有条件都满足，就是异常，有一个不满足就是正常
        tag = True
        for k in range(len(feature_lst)):
            feature = feature_lst[k]
            inequality = inequality_lst[k]
            threshold = threshold_lst[k]
            if inequality == '<':
                if data[j][feature] < threshold:
                    pass
                else:
                    tag = False
                    break

            elif inequality == '>':
                if data[j][feature] > threshold:
                    pass
                else:
                    tag = False
                    break

            elif inequality == '>=':
                if data[j][feature] >= threshold:
                    pass
                else:
                    tag = False
                    break

            elif inequality == '<=':
                if data[j][feature] <= threshold:
                    pass
                else:
                    tag = False
                    break

        if tag == True:
            pred_array[j] = 1
        else:
            pred_array[j] = 0

    return pred_array

# def stream_normal_rule_evaluate(args, config,
#                                 explain_instance,
#                                 explained_instance_label,
#                                 explained_instance_idx,
#                                 explained_data,
#                                 explained_data_label,
#                                 feature_imp,
#                                 key_points_lgp_correctify,
#                                 core_point_inequality_correctify,
#                                 proto_path,
#                                 data_use,
#                                 label_use
#                                 ):
#     batch_idx = int(explained_instance_idx / config.batch_size) - 1
#
#     proto_name = f'batchidx_{batch_idx}.npy'
#     normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
#
#     # idx_lst = sum([proto.data_idx_lst for proto in normal_protos], [])
#     idx_lst = []
#     for proto in normal_protos:
#         random_idx_lst = list(
#             np.random.choice(proto.data_idx_lst, int(len(proto.data_idx_lst) / 10), replace=False))
#         idx_lst.extend(random_idx_lst)
#
#     idx_array = np.array(idx_lst)
#     center_array = np.array([proto.center for proto in normal_protos])
#     r_array = np.array([proto.r for proto in normal_protos])
#
#     attacks = []
#     explained_data_same_label = explained_data[np.where(explained_data_label == explained_instance_label)[0]]
#     dist_matrix = MatrixEuclideanDistances(explained_data_same_label, center_array)
#     for j in range(len(dist_matrix)):
#         cover_idx = np.where(dist_matrix[j] <= r_array ** 2 + args.eps)[0]
#         if len(cover_idx) == 0:
#             attacks.append(explained_data_same_label[j])
#     attacks = np.array(attacks)
#
#     pass_data = data_use[idx_array]
#     pass_label = label_use[idx_array]
#     normal_idx = np.where(pass_label == 0)[0]
#
#     data_metric_array = pass_data[normal_idx]
#
#     data = np.vstack((data_metric_array, attacks))
#     label = np.array(list(np.zeros(len(data_metric_array))) + list(np.ones(len(attacks))))
#
#
#     pred_array = np.zeros(len(data))
#     for j in range(len(data)):
#         check_instance = data[j]
#         pred_array[j] = rule_compared(check_instance, key_points_lgp_correctify, core_point_inequality_correctify,
#                                       feature_imp)
#
#     roc_value = roc_auc(label, pred_array)
#     pr_value = pr(label, pred_array)
#     f1_value = f1_score(label, pred_array, average='macro')
#     cm = confusion_matrix(y_true=label, y_pred=pred_array, labels=sorted(list(np.unique(label))))
#     attack_recall_value = cm[1, 1] / np.sum(cm[1])
#
#     return roc_value, pr_value, f1_value, attack_recall_value
#
#
# @nb.njit()
# def rule_compared(check_instance, core_points, inequality_all, feature_imp):
#     pred = 0
#     outside_num = 0
#     for i in range(len(core_points)):
#         threshold = core_points[i]
#         inequality = inequality_all[i]
#         each_grid_tag = "inside"
#         for k in range(len(feature_imp)):
#             feature_imp_idx = feature_imp[k]
#             if (check_instance[feature_imp_idx] <= threshold[feature_imp_idx] and inequality[k] == '<') or \
#                     (check_instance[feature_imp_idx] >= threshold[feature_imp_idx] and inequality[k] == '>='):
#                 each_grid_tag = "outside"
#             else:
#                 pass
#
#         if each_grid_tag == "inside":
#             break
#         else:
#             outside_num += 1
#
#     if outside_num == len(core_points):
#         pred = 1
#     return pred


def stream_feature_evaluate_ids(args, config, f_imp_array, explained_idx_in_attacks_array,
                                fea_num_dict, explain_data, explain_label, explain_idx, proto_path, data_use,
                                label_use):
    explained_data = explain_data[explained_idx_in_attacks_array]
    explained_label = explain_label[explained_idx_in_attacks_array]
    explained_idx = explain_idx[explained_idx_in_attacks_array]

    assert len(explained_data) == len(f_imp_array)

    in_circle_feat_lst = []
    imp_fea_num_feat_lst = []

    # feature_num_t = feature_num_dict[args.feat]


    feature_num_array = fea_num_dict[args.feat]
    in_circle_ratio = Parallel(n_jobs=-1)(
        delayed(stream_feature_evaluate_ids_each_instance)(args, config,
                                                           explained_data[i],
                                                           explained_idx[i],
                                                           f_imp_array[i],
                                                           proto_path,
                                                           data_use,
                                                           label_use,
                                                           feature_num_array[i]
                                                           ) for i in range(len(explained_data)))

    in_circle_feat_lst.append(np.mean(in_circle_ratio))
    # imp_fea_num_feat_lst.append(feature_num_t)
    imp_fea_num_feat_lst.append(np.mean(feature_num_array))

    return in_circle_feat_lst, imp_fea_num_feat_lst


def stream_feature_evaluate_ids_each_instance(args, config,
                                              explained_instance,
                                              explained_instance_idx,
                                              feature_importance,
                                              proto_path,
                                              data_use, label_use,
                                              feature_num_t):
    batch_idx = int(explained_instance_idx / config.batch_size) - 1

    proto_name = f'batchidx_{batch_idx}.npy'
    normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)

    center_array = np.array([proto.center for proto in normal_protos])
    r_array = np.array([proto.r for proto in normal_protos])

    # idx_lst = sum([proto.data_idx_lst for proto in normal_protos], [])
    idx_lst = []
    for proto in normal_protos:
        random_idx_lst = list(
            np.random.choice(proto.data_idx_lst, int(len(proto.data_idx_lst) / 10), replace=False))
        idx_lst.extend(random_idx_lst)

    idx_array = np.array(idx_lst)

    pass_data = data_use[idx_array]
    pass_label = label_use[idx_array]
    normal_idx = np.where(pass_label == 0)[0]
    data_metric_array = pass_data[normal_idx]

    # if args.exp_method == 'Enids':
    #     imp_idx = np.where(feature_importance >= fea_t - args.eps)[0]
    #     if len(imp_idx) == 0:
    #         imp_idx = np.array([np.argmax(feature_importance)])
    if args.exp_method == 'random':
        imp_idx = np.random.choice(len(explained_instance), feature_num_t, replace=False)
    else:
        imp_idx = np.argsort(-feature_importance)[:feature_num_t]

    replace = np.zeros(len(feature_importance))
    replace[imp_idx] = 1
    remain = 1 - replace
    data_metric_new = replace * data_metric_array + remain * explained_instance
    dist_matrix = MatrixEuclideanDistances(data_metric_new, center_array)

    in_circle_num = find_in_circle(data_metric_new, dist_matrix, r_array, args.eps)

    return in_circle_num / len(data_metric_new)


@nb.njit()
def find_in_circle(data_metric_new, dist_matrix, r_array, eps):
    in_circle_num = 0
    for j in range(len(data_metric_new)):
        cover_idx = np.where(dist_matrix[j] <= r_array ** 2 + eps)[0]
        if len(cover_idx) != 0:
            in_circle_num += 1

    return in_circle_num