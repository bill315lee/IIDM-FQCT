from explain_method.model_acekl.ace_explainer_pt import AceExplainer
from collections import Counter
from tqdm import tqdm
import numpy as np


class ACE_KL:
    def __init__(self):

        self.epochs = 1000


    def fit(self, all_data, all_label, explain_instances):

        f_imp_lst = []
        for instance in tqdm(explain_instances):
            acekl_explner = AceExplainer(instance, len(instance), self.epochs, is_usekl=True)
            acekl_exp = acekl_explner.explain_instance(instance, all_data, all_label)
            weight_array = np.zeros(len(instance))
            for x in acekl_exp.as_map():
                weight_array[x[0]] = x[1]
            f_imp_lst.append(weight_array)

        return f_imp_lst