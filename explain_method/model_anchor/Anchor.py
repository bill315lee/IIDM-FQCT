import numpy as np
import re
from utils import predictor_train, str_to_bool
rule_key = re.compile('^[a-zA-z]{1}.*$')
from joblib import Parallel, delayed
from explain_intro.model_anchor import utils
from explain_intro.model_anchor import anchor_tabular




class Anchor_rule:
    def __init__(self, args, feature_names, config, kernel="rbf"):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.threshold = args.t_anchor
        self.kernel = kernel
        self.batch_size = config.batch_size
        self.eps = args.eps
        self.feature_names = feature_names

        self.train_all_data = str_to_bool(args.train_all_data)
        self.attack_upper_num = args.attack_upper_num



    def anchor_rule_generation(self,explain_instance, explain_instance_label,explain_instance_idx, explain_instances_class, proto_path, rule_num_instance, data_use):

        normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
        center_array = np.array([proto.center for proto in normal_protos])

        if self.train_all_data:
            idx_lst = sum([proto.data_idx_lst for proto in normal_protos], [])
            idx_array = np.array(idx_lst)
            pass_data = data_use[idx_array]

            inputs = np.vstack((pass_data, explain_instances_class, explain_instance.reshape(1, -1)))
            targets = np.array(list(np.zeros(len(pass_data))) + list(np.ones(len(explain_instances_class))) + [1], dtype=int)
            predictor = predictor_train(inputs, targets)
        else:
            inputs = np.vstack((center_array, explain_instances_class, explain_instance.reshape(1, -1)))
            targets = np.array(list(np.zeros(len(center_array))) + list(np.ones(len(explain_instances_class))) + [1], dtype=int)
            predictor = predictor_train(inputs, targets)


        explainer = anchor_tabular.AnchorTabularExplainer([0,1], self.feature_names, inputs)

        explanation = explainer.explain_instance(explain_instance.reshape(1, -1), predictor.predict, threshold=0.95)


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


    def fit(self, explain_data, explain_label, explain_idx, proto_path, rule_num, data_use):






        attack_dict = {}
        for label_key in set(explain_label):
            explain_instances_class = explain_data[np.where(explain_label == label_key)[0]]
            attack_dict[label_key] = explain_instances_class

        results = Parallel(n_jobs=-1)(
            delayed(self.anchor_rule_generation)(
                                          explain_data[i],
                                          explain_label[i],
                                          explain_idx[i],
                                          attack_dict[explain_label[i]],
                                          proto_path,
                                            rule_num[i],
                                          data_use) for i in range(len(explain_data)))

        feature_lst_alldata = []
        inequality_lst_alldata = []
        threshold_lst_alldata = []
        for i in range(len(results)):
            feature_lst_alldata.append(results[i][0])
            inequality_lst_alldata.append(results[i][1])
            threshold_lst_alldata.append(results[i][2])




        return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata

