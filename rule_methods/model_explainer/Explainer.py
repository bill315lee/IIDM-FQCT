
import numpy as np
from rule_methods.model_explainer.Explainer_Forest import Explainer_Forest
from tqdm import tqdm

class Explainer:
    def __init__(self, tree_nums, sample_size):

        self.tree_nums = tree_nums
        self.sample_size = sample_size

    def fit(self, explain_data, explain_label, normal_protos, data_use):
        self.dim = explain_data.shape[1]

        feature_lst_alldata = []
        inequality_lst_alldata = []
        threshold_lst_alldata = []
        rule_num = []
        for i in tqdm(range(len(explain_data))):

            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]


            ex_forest = Explainer_Forest(sample_size=self.sample_size, n_estimators=self.tree_nums)
            ex_forest.fit(data_use, np.zeros(len(data_use)), explain_instance, explain_instance_label)

            H_subspace = {}
            for t in ex_forest.trees:
                inequality_lst = t.inequality_lst
                feature_lst = t.feature_lst
                threshold_lst = t.threshold_lst

                for j in range(len(inequality_lst)):
                    inequality = inequality_lst[j]
                    feature = feature_lst[j]
                    threshold = threshold_lst[j]
                    tag = str(feature) + '_' + inequality
                    if tag in H_subspace:
                        H_subspace[tag].append(threshold)
                    else:
                        H_subspace[tag] = [threshold]

            rule_tags = np.array(list(H_subspace.keys()))
            rule_thresholds = np.array(list(H_subspace.values()))

            rule_nums = np.array([len(rule) for rule in rule_thresholds])

            all_rule_nums = np.sum(rule_nums)

            sum_k_rules = 0
            idx_rules = []
            for idx in np.argsort(-rule_nums):
                sum_k_rules += rule_nums[idx]
                idx_rules.append(idx)
                if sum_k_rules / all_rule_nums > 0.9:
                    break
            rule_choose = rule_tags[idx_rules]
            rule_thresholds_choose = rule_thresholds[idx_rules]



            feature_lst = []
            inequality_lst = []
            threshold_lst = []

            for j in range(len(rule_choose)):
                tag = rule_choose[j]
                feature_lst.append(int(tag.split('_')[0]))
                inequality_lst.append(tag.split('_')[1])
                threshold_lst.append(np.median(rule_thresholds_choose[j]))

            # print(len(feature_lst), rule_eids_num[i])
            # if len(set(feature_lst)) > rule_eids_num[i]:
            #     if len(set(feature_lst)) == len(feature_lst):
            #         feature_lst = feature_lst[:rule_eids_num[i]]
            #         inequality_lst = inequality_lst[:rule_eids_num[i]]
            #         threshold_lst = threshold_lst[:rule_eids_num[i]]
            #     else:
            #         idx_lst_new = []
            #         fea_lst_new = []
            #         for j in range(len(feature_lst)):
            #             if len(set(fea_lst_new + [feature_lst[j]])) > rule_eids_num[i]:
            #                 break
            #             else:
            #                 fea_lst_new.append(feature_lst[j])
            #                 idx_lst_new.append(j)
            #
            #         inequality_lst = list(np.array(inequality_lst)[idx_lst_new])
            #         threshold_lst = list(np.array(threshold_lst)[idx_lst_new])
            #         feature_lst = fea_lst_new

            # print(len(feature_lst), rule_eids_num[i])
            # print('-----------------')

            # print('explained_instance=', explain_instance[np.array(feature_lst)])
            # print('normal_max=', np.max(center_array[:,np.array(feature_lst)], axis=0))
            # print('normal_min=', np.min(center_array[:,np.array(feature_lst)], axis=0))
            #
            # print('feature_lst=', feature_lst)
            # print('ineq_lst=', inequality_lst)
            # print('thre_lst=', threshold_lst)

            feature_lst_alldata.append(feature_lst)
            inequality_lst_alldata.append(inequality_lst)
            threshold_lst_alldata.append(threshold_lst)
            rule_num.append(len(set(feature_lst)))

        return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata, rule_num



