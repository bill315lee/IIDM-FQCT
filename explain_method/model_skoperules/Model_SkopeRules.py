
from skrules import SkopeRules
import numpy as np
from tqdm import tqdm

class Skoprules:

    def __init__(self, feature_names):

        self.feature_names = feature_names

    def explain(self, data_test, label_test, data_use, feature_num_array):

        attack_dict = {}
        feature_all_instance = []
        inequality_all_instance = []
        threshold_all_instance = []

        for i in tqdm(range(len(data_test))):
            feature_num = feature_num_array[i]
            explain_instance = data_test[i]


            clf = SkopeRules(feature_names=self.feature_names,
                             random_state=0)

            if label_test[i] in attack_dict:
                attack_data = attack_dict[label_test[i]][-3:]
                data = np.vstack((data_use, attack_data, explain_instance.reshape(1,-1)))
                label = np.array(list(np.zeros(len(data_use))) + list(np.ones(len(attack_data))) + [1])
            else:
                data = np.vstack((data_use, explain_instance.reshape(1, -1)))
                label = np.array(list(np.zeros(len(data_use))) + [1])

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
                threshold = np.random.uniform(np.min(data_use[feature_idx]), np.max(data_use[feature_idx]))
                feature_lst_each_instance.append([feature_idx])
                threshold_lst_each_instance.append([threshold])
                ineq_lst_each_instance.append(['<'])


            feature_all_instance.append(feature_lst_each_instance[0])
            inequality_all_instance.append(ineq_lst_each_instance[0])
            threshold_all_instance.append(threshold_lst_each_instance[0])

            if label_test[i] in attack_dict:
                attack_dict[label_test[i]].append(data_test[i])
            else:
                attack_dict[label_test[i]] = [data_test[i]]


            # print('feature_lst=',feature_lst_each_instance[0])
            # print('ineq_lst=',ineq_lst_each_instance[0])
            # print('threshold_lst=',threshold_lst_each_instance[0])



            
        return feature_all_instance, inequality_all_instance, threshold_all_instance



