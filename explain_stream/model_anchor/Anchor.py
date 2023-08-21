import sklearn
import numpy as np
from alibi.explainers import AnchorTabular
from tqdm import tqdm
from collections import Counter
import re
from utils import euclidian_dist, predictor_train
rule_key = re.compile('^[a-zA-z]{1}.*$')




class Anchor_rule:
    def __init__(self,args, config, kernel="rbf"):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.threshold = args.t_anchor
        self.kernel = kernel
        self.batch_size = config.batch_size
        self.dim = None
        self.attack_dict = {}
        self.eps = args.eps


    def fit(self, explain_data, explain_label, explain_idx, proto_path, rule_num, feature_names):


        feature_lst_alldata = []
        inequality_lst_alldata = []
        threshold_lst_alldata = []

        for i in tqdm(range(len(explain_data))):
            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]
            explain_instance_idx = explain_idx[i]
            batch_idx = int(explain_instance_idx / self.batch_size) - 1

            proto_name = f'batchidx_{batch_idx}.npy'
            normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)
            center_array = np.array([proto.center for proto in normal_protos])


            if explain_instance_label in self.attack_dict:
                attacks = np.array(self.attack_dict[explain_instance_label])
                attack_labels = np.ones(len(self.attack_dict[explain_instance_label]))

                inputs = np.vstack((center_array, attacks, explain_instance.reshape(1, -1)))
                targets = np.array(list(np.zeros(len(center_array))) + list(attack_labels) + [1], dtype=int)
                sizes = np.array(list(np.ones(len(center_array))) + list(attack_labels) + [1])
            else:
                inputs = np.vstack((center_array, explain_instance.reshape(1, -1)))
                targets = np.array(list(np.zeros(len(center_array))) + [1], dtype=int)
                sizes = np.array(list(np.ones(len(center_array))) + [1])

            predictor = predictor_train(inputs, targets, sizes)
            predict_fn = lambda xx: predictor.predict_proba(xx)


            explainer = AnchorTabular(predict_fn, feature_names)
            explainer.fit(inputs, disc_perc=(25, 50, 75))

            explanation = explainer.explain(explain_instance.reshape(1,-1), threshold=self.threshold)


            feature_lst = []
            inequality_lst = []
            threshold_lst = []
            anchor_exp = explanation['anchor'][:rule_num[i]]
            for anchor in anchor_exp:
                rule = anchor.split(" ")
                if len(rule) > 3:
                    for item in rule:
                        if rule_key.match(item):
                            break
                    idx = rule.index(item)
                    threshold_inequality_former = rule[:idx]
                    threshold_inequality_latter = rule[idx + 1:]


                    feature_lst.append(feature_names.index(item))
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

                    feature_lst.append(feature_names.index(item))
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
                            feature_lst.append(feature_names.index(item))
                        elif item in ['>', '<', '>=', '<=']:
                            inequality_lst.append(item)
                        else:
                            threshold_lst.append(float(item))


            feature_lst_alldata.append(feature_lst)
            inequality_lst_alldata.append(inequality_lst)
            threshold_lst_alldata.append(threshold_lst)





        return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata

