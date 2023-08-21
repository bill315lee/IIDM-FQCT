
import numpy as np
from explain_intro.model_explainer.Explainer_Forest import Explainer_Forest
from tqdm import tqdm
from utils import euclidian_dist, str_to_bool


class Explainer:
    def __init__(self, args, config):

        self.tree_nums = args.treesnum_explainer
        self.samples = args.sample_explainer
        self.eps = args.eps
        self.batch_size = config.batch_size
        self.train_all_data = str_to_bool(args.train_all_data)

        self.attack_upper_num = args.attack_upper_num
    def fit(self, explain_data, explain_label, explain_idx, proto_path, data_use):


        feature_lst_alldata = []
        inequality_lst_alldata = []
        threshold_lst_alldata = []
        rule_num = []

        for i in tqdm(range(len(explain_data))):
            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]
            explain_instance_idx = explain_idx[i]
            batch_idx = int(explain_instance_idx / self.batch_size) - 1
            proto_name = f'batchidx_{batch_idx}.npy'
            normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
            proto_array = np.array([proto for proto in normal_protos])
            center_array = np.array([proto.center for proto in normal_protos])

            # sample_size = np.min([self.samples, len(center_array)])
            # sample_size = int(0.3 * len(center_array))

            ex_forest = Explainer_Forest(sample_size=self.samples, n_estimators=self.tree_nums)

            if self.train_all_data:
                idx_lst = sum([proto.data_idx_lst for proto in proto_array], [])
                idx_array = np.array(idx_lst)
                pass_data = data_use[idx_array]
                ex_forest.fit(pass_data, np.zeros(len(pass_data)), explain_instance, explain_instance_label)
                print(pass_data.shape)
            else:
                ex_forest.fit(center_array, np.zeros(len(center_array)), explain_instance, explain_instance_label)
                print(center_array.shape)


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


            # if len(set(feature_lst)) > rule_eids_num[i]:
            #     if len(set(feature_lst)) == len(feature_lst):
            #         feature_lst = feature_lst[:rule_eids_num[i]]
            #         inequality_lst = inequality_lst[:rule_eids_num[i]]
            #         threshold_lst = threshold_lst[:rule_eids_num[i]]
            #     else:
            #         idx_lst_new = []
            #         fea_lst_new = []
            #         for j in range(len(feature_lst)):
            #             if len(set(fea_lst_new+[feature_lst[j]])) > rule_eids_num[i]:
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
            # for fea in feature_num_dict:
            #     feature_tmp = []
            #     tag = False
            #     for j in range(len(feature_lst)):
            #         feature_tmp.append(feature_lst[j])
            #         if len(set(feature_tmp)) == feature_num_dict[fea][i]:
            #             feature_all_fea.append(feature_lst[:j + 1])
            #             inequality_all_fea.append(inequality_lst[:j + 1])
            #             threshold_all_fea.append(threshold_lst[:j + 1])
            #             tag = True
            #             break
            #     if tag == False:
            #         feature_all_fea.append(feature_lst)
            #         inequality_all_fea.append(inequality_lst)
            #         threshold_all_fea.append(threshold_lst)

            # print('final')
            # print(feature_lst)
            # print(inequality_lst)
            # print(threshold_lst)

            feature_lst_alldata.append(feature_lst)
            inequality_lst_alldata.append(inequality_lst)
            threshold_lst_alldata.append(threshold_lst)
            rule_num.append(len(set(feature_lst)))

            if len(feature_lst_alldata) == self.attack_upper_num:
                break




        return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata, rule_num



