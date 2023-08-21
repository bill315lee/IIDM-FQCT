import lime
import lime.lime_tabular
import numpy as np
import sklearn
import math
from tqdm import tqdm
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
class LIME:
    def __init__(self, discretize_continuous=True, discretizer='quartile'):

        self.discretize_continuous = discretize_continuous
        self.discretizer = discretizer

        self.dim = None
        self.ano_idx = None
        return

    def fit(self, x, y, explain_data, explain_label, predictor):

        self.dim = x.shape[1]

        explainer = lime.lime_tabular.LimeTabularExplainer(x, discretize_continuous=self.discretize_continuous, discretizer=self.discretizer)

        ano_f_weights = np.zeros([len(explain_data), self.dim])

        for i in tqdm(range(len(explain_data))):
            exp = explainer.explain_instance(explain_data[i], predictor.predict_proba, labels=(explain_label[i],), num_features=self.dim)
            tuples = exp.as_map()[explain_label[i]]
            for tuple in tuples:
                f_idx, weight = tuple
                ano_f_weights[i][f_idx] = weight

        return ano_f_weights



