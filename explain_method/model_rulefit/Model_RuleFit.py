from rulefit import RuleFit
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np


class RuleFit_Model:

    def __init__(self, feature_names):

        self.feature_names = feature_names

    def explain(self, data_test, label_test, proto_array, feature_num_array):

        center_array = np.array([proto.center for proto in proto_array])
        attack_dict = {}

        feature_all_instance = []
        inequality_all_instance = []
        threshold_all_instance = []

        for i in range(len(data_test)):
            feature_num = feature_num_array[i]
            explain_instance = data_test[i]
            # if label_test[i] in attack_dict:
            #     attack_dict[label_test[i]].append(data_test[i])
            # else:
            #     attack_dict[label_test[i]] = [data_test[i]]

            gb = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.01)
            rf = RuleFit(tree_generator = gb)

            data = np.vstack((center_array, explain_instance.reshape(1,-1)))
            label = np.array(list(np.zeros(len(center_array))) + [1])

            rf.fit(data, label, feature_names=self.feature_names)

            importance = rf.get_rules().values[len(self.feature_names):,-1]
            important_rules = rf.get_rules().values[len(self.feature_names):,0][np.argmax(importance)]

            # print(important_rules)
            feature_lst_each_instance = []
            ineq_lst_each_instance = []
            threshold_lst_each_instance = []

            if '&' in important_rules:
                rule_combine = important_rules.split('&')
                for subrule in rule_combine:
                    feature_ineq_threshold = subrule.split(' ')
                    # print(rule_combine, subrule, feature_ineq_threshold)
                    while "" in feature_ineq_threshold:
                        feature_ineq_threshold.remove("")

                    ineq_lst_each_instance.append(feature_ineq_threshold[1])
                    feature_lst_each_instance.append(self.feature_names.index(feature_ineq_threshold[0]))
                    threshold_lst_each_instance.append(float(feature_ineq_threshold[2]))
            else:

                feature_ineq_threshold = important_rules.split(' ')
                ineq_lst_each_instance.append(feature_ineq_threshold[1])
                feature_lst_each_instance.append(self.feature_names.index(feature_ineq_threshold[0]))
                threshold_lst_each_instance.append(float(feature_ineq_threshold[2]))


            # print(feature_lst_each_instance)
            # print(ineq_lst_each_instance)
            # print(threshold_lst_each_instance)

            feature_all_instance.append(feature_lst_each_instance)
            inequality_all_instance.append(ineq_lst_each_instance)
            threshold_all_instance.append(threshold_lst_each_instance)


        return feature_all_instance, inequality_all_instance, threshold_all_instance


