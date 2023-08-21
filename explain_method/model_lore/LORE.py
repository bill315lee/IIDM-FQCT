import sklearn
import numpy as np
from alibi.explainers import AnchorTabular
from tqdm import tqdm
from explain_method.model_lore import pyyadt
import random

from explain_method.model_lore.neighbor_generator import *
from explain_method.model_lore.gpdatagenerator import calculate_feature_values
from explain_method.model_lore.prepare_dataset import prepare_dataset

class Lore:
    def __init__(self,args):

        self.args = args


    def fit(self, data_train, label_train, explain_data, explain_label, predictor, feature_names):


        x = np.vstack((data_train, explain_data))
        y = np.array(list(label_train) + list(explain_label), dtype=int)
        data_label = np.hstack((x, y.reshape(-1, 1)))

        df = pd.DataFrame(data=data_label, columns=feature_names + ['class'])
        dataset = prepare_dataset(df, self.args)
        path_data = '/home/lb/project/explain_ids/pathdata/'
        lore_exp_lst = []
        for i in range(len(explain_data)):

            explanation, infos = lore_explain(i, explain_data, dataset, predictor,
                                              ng_function=genetic_neighborhood,
                                              discrete_use_probabilities=True,
                                              continuous_function_estimation=False,
                                              returns_infos=True,
                                              path=path_data, sep=';', log=False)
            print('i=',i, explanation[0][1])
            exit()
            lore_exp_lst.append(explanation[0][1])
            # print("explanation=", explanation[0][1])


        return lore_exp_lst




def lore_explain(idx_record2explain, data_test, dataset, blackbox,
            ng_function=genetic_neighborhood, #generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False, path='./', sep=';', log=False):

    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    possible_outcomes = dataset['possible_outcomes']

    # Dataset Preprocessing
    # dataset['feature_values'] = calculate_feature_values(data_test, columns, class_name, discrete, continuous, 1000, discrete_use_probabilities, continuous_function_estimation)
    dataset['feature_values'] = data_test

    dfZ, x = dataframe2explain(data_test, dataset, idx_record2explain, blackbox)

    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous, filename=dataset['name'], path=path, sep=sep, log=log)

    # Apply Black Box and Decision Tree on instance to explain
    bb_outcome = blackbox.predict(x.reshape(1, -1))[0]

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # # Update labels if necessary
    # if class_name in label_encoder:
    #     cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]
    #
    # if class_name in label_encoder:
    #     y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome, class_name, continuous, features_type)
    print("counterfactuals=", counterfactuals)
    explanation = (rule, counterfactuals)

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome,
        'predict': predict,
    }

    if returns_infos:
        return explanation, infos

    return explanation


def is_satisfied(x, rule, discrete, features_type):
    for col, val in rule.items():
        if col in discrete:
            if str(x[col]).strip() != val:
                return False
        else:
            if '<=' in val and '<' in val and val.find('<=') < val.find('<'):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<=', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] > thr1 or x[col] <= thr2:
                    return False
            elif '<' in val and '<=' in val and val.find('<') < val.find('<='):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<=', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] >= thr1 or x[col] < thr2:
                    return False
            elif '<=' in val:
                thr = pyyadt.yadt_value2type(val.replace('<=', ''), col, features_type)
                if x[col] > thr:
                    return False
            elif '>' in val:
                thr = pyyadt.yadt_value2type(val.replace('>', ''), col, features_type)
                if x[col] <= thr:
                    return False
    return True


def get_covered(rule, X, dataset):
    covered_indexes = list()
    for i, x in enumerate(X):
        if is_satisfied(x, rule, dataset['discrete'], dataset['features_type']):
            covered_indexes.append(i)
    return covered_indexes





