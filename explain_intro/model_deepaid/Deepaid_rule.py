import numpy as np
from tqdm import tqdm


def Deepaid_rule_generation(f_imp_array, rule_num, explained_data):


    feature_all_instance = []
    inequality_all_instance = []
    threshold_all_instance = []


    for i in tqdm(range(len(explained_data))):

        explain_instance = explained_data[i]

        inequality_instance = []
        threshold_instance = []
        counterfactual = f_imp_array[i]
        for j in range(len(counterfactual)):
            if explain_instance[j] < counterfactual[j]:
                inequality_instance.append('<')
            else:
                inequality_instance.append('>=')
            threshold_instance.append(counterfactual[j])

        inequality_instance_array = np.array(inequality_instance)
        threshold_instance_array = np.array(threshold_instance)


        idx = np.argsort(-counterfactual)[:rule_num[i]]


        feature_all_instance.append(idx)
        inequality_all_instance.append(inequality_instance_array[idx])
        threshold_all_instance.append(threshold_instance_array[idx])



    return feature_all_instance, inequality_all_instance, threshold_all_instance