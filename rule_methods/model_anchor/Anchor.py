import numpy as np
import re
from utils import predictor_train, str_to_bool
rule_key = re.compile('^[a-zA-z]{1}.*$')
from joblib import Parallel, delayed
from explain_intro.model_anchor import utils
from explain_intro.model_anchor import anchor_tabular
from tqdm import tqdm



class Anchor_rule:
    def __init__(self, args, feature_names, kernel="rbf"):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.threshold = args.t_anchor
        self.kernel = kernel
        self.eps = args.eps
        self.feature_names = feature_names


    def anchor_rule_generation(self, explained_instance, explain_instance_label, attack_pass_data, data_use, rule_num_instance, size_lst):


        data = np.vstack((data_use, attack_pass_data))
        label = np.array(list(np.zeros(len(data_use))) + list(np.ones(len(attack_pass_data))))
        size = np.array(size_lst + list(np.ones(len(attack_pass_data))))
        predictor = predictor_train(data, label, sample_weight=size)

        explainer = anchor_tabular.AnchorTabularExplainer([0,1], self.feature_names, data)
        explanation = explainer.explain_instance(explained_instance.reshape(1, -1), predictor.predict, threshold=0.95)


        feature_lst = []
        inequality_lst = []
        threshold_lst = []
        anchor_exp = explanation.exp_map['names'][:rule_num_instance]
        # anchor_exp = explanation.exp_map['names']
        for anchor in anchor_exp:
            rule = anchor.split(" ")
            if len(rule) > 3:
                for item in rule:
                    if rule_key.match(item):
                        break
                idx = rule.index(item)
                threshold_inequality_former = rule[:idx]
                threshold_inequality_latter = rule[idx + 1:]

                feature_lst.append(self.feature_names.index(item))
                for key in threshold_inequality_former:
                    if key == '>':
                        inequality_lst.append('<')
                    elif key == '<':
                        inequality_lst.append('>=')
                    elif key == '<=':
                        inequality_lst.append('>=')
                    elif key == '>=':
                        inequality_lst.append('<')
                    else:
                        threshold_lst.append(float(key))

                feature_lst.append(self.feature_names.index(item))
                for key in threshold_inequality_latter:
                    if key in ['>', '<', '>=', '<=']:
                        if key == '>' or key == '>=':
                            inequality_lst.append('>=')
                        elif key == '<' or key == '<=':
                            inequality_lst.append('<')
                        else:
                            raise NotImplementedError
                    else:
                        threshold_lst.append(float(key))

            else:
                for item in rule:
                    if rule_key.match(item):
                        feature_lst.append(self.feature_names.index(item))
                    elif item in ['>', '<', '>=', '<=']:
                        inequality_lst.append(item)
                    else:
                        threshold_lst.append(float(item))

        return feature_lst, inequality_lst, threshold_lst


    def fit(self, data_test, label_test, feature_num_array, data_use, size_lst):


        attack_dict = {}
        attack_pass_data = []
        for i in tqdm(range(len(data_test))):

            if label_test[i] in attack_dict:
                attack_dict[label_test[i]].append(data_test[i])
                attack_pass_data.append(attack_dict[label_test[i]])
            else:
                attack_dict[label_test[i]] = [data_test[i]]
                attack_pass_data.append(attack_dict[label_test[i]])


        results = Parallel(n_jobs=-1)(
            delayed(self.anchor_rule_generation)(data_test[i],
                                                 label_test[i],
                                                 attack_pass_data[i],
                                                 data_use,
                                                 feature_num_array[i],
                                                 size_lst
                                                 ) for i in range(len(data_test)))
        feature_lst_alldata = []
        inequality_lst_alldata = []
        threshold_lst_alldata = []

        for i in range(len(results)):
            feature_lst_alldata.append(results[i][0])
            inequality_lst_alldata.append(results[i][1])
            threshold_lst_alldata.append(results[i][2])

            print(results[i][0],results[i][1], results[i][2])







        return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata

