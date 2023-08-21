
from skrules import SkopeRules
import numpy as np
from tqdm import tqdm

class Skoprules:

    def __init__(self, feature_names, batch_size):

        self.feature_names = feature_names

        self.batch_size = batch_size

    def explain(self,
                explained_data_all,
                explained_label_all,
                explained_idx_all,
                proto_path):

        attack_dict = {}


        feature_all_instance = []
        inequality_all_instance = []
        threshold_all_instance = []

        for i in tqdm(range(len(explained_data_all))):

            explain_instance = explained_data_all[i]
            explain_instance_label = explained_label_all[i]
            explain_instance_idx = explained_idx_all[i]

            batch_idx = int(explain_instance_idx / self.batch_size) - 1
            proto_name = f'batchidx_{batch_idx}.npy'
            normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
            center_array = np.array([proto.center for proto in normal_protos])

            clf = SkopeRules(n_estimators=20, feature_names=self.feature_names, random_state=0)

            if explain_instance_label in attack_dict:
                attack_data = np.array(attack_dict[explain_instance_label][-2:])
                data = np.vstack((center_array, attack_data, explain_instance.reshape(1, -1)))
                label = np.array(list(np.zeros(len(center_array))) + list(np.ones(len(attack_data))) + [1])
            else:
                data = np.vstack((center_array, explain_instance.reshape(1, -1)))
                label = np.array(list(np.zeros(len(center_array))) + [1])

            clf.fit(data, label)

            rules = clf.rules_

            rul_lst = [rule[0] for rule in rules]



            feature_lst_each_instance = []
            ineq_lst_each_instance = []
            threshold_lst_each_instance = []
            for rule in rul_lst:
                feature_lst = []
                ineq_lst = []
                threshold_lst = []

                if 'and' in rule:
                    rule_combine = rule.split('and')
                    for subrule in rule_combine:
                        feature_ineq_threshold = subrule.split(' ')
                        # print(rule_combine, subrule, feature_ineq_threshold)
                        while "" in feature_ineq_threshold:
                            feature_ineq_threshold.remove("")
                        if feature_ineq_threshold[1] == '==':
                            continue
                        ineq_lst.append(feature_ineq_threshold[1])
                        feature_lst.append(self.feature_names.index(feature_ineq_threshold[0]))
                        threshold_lst.append(float(feature_ineq_threshold[2]))

                else:
                    feature_ineq_threshold = rule.split(' ')
                    if feature_ineq_threshold[1] == '==':
                        continue
                    ineq_lst.append(feature_ineq_threshold[1])
                    feature_lst.append(self.feature_names.index(feature_ineq_threshold[0]))
                    threshold_lst.append(float(feature_ineq_threshold[2]))

                feature_lst_each_instance.append(feature_lst)
                ineq_lst_each_instance.append(ineq_lst)
                threshold_lst_each_instance.append(threshold_lst)

            if len(feature_lst_each_instance) == 0:
                feature_idx = np.random.choice(np.arange(len(self.feature_names)), 1)[0]
                threshold = np.random.uniform(np.min(center_array[feature_idx]), np.max(center_array[feature_idx]))
                feature_lst_each_instance.append([feature_idx])
                threshold_lst_each_instance.append([threshold])
                ineq_lst_each_instance.append(['<'])


            feature_all_instance.append(feature_lst_each_instance[0])
            inequality_all_instance.append(ineq_lst_each_instance[0])
            threshold_all_instance.append(threshold_lst_each_instance[0])

            # print('feature_lst=',feature_lst_each_instance[0])
            # print('ineq_lst=',ineq_lst_each_instance[0])
            # print('threshold_lst=',threshold_lst_each_instance[0])

            if explain_instance_label in attack_dict:
                attack_dict[explain_instance_label].append(explain_instance)
            else:
                attack_dict[explain_instance_label] = [explain_instance]


            
        return feature_all_instance, inequality_all_instance, threshold_all_instance



