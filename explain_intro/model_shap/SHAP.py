import shap
import math
import random
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from tqdm import tqdm


class SHAP:
    def __init__(self, kernel="rbf", n_sample=100, threshold=0.8):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.kernel = kernel
        self.threshold = threshold
        self.n_sample = n_sample
        return

    def fit(self, x, y, explain_instance, explain_label, predictor):


        x_samples = shap.sample(x, self.n_sample)
        explainer = shap.KernelExplainer(predictor.predict_proba, x_samples)

        f_imp = explainer.shap_values(explain_instance)[explain_label][0]

        return f_imp


