import shap
import numpy as np
from tqdm import tqdm


class SHAP:
    def __init__(self, kernel="rbf", n_sample=50, threshold=0.8):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """

        self.kernel = kernel
        self.threshold = threshold
        self.n_sample = n_sample
        self.dim = None
        return

    def fit(self, x, y, explain_data, explain_label, predictor):
        f_imp = []
        self.dim = x.shape[1]
        # x_kmean = shap.kmeans(x, self.n_sample)

        x_samples = shap.sample(x, self.n_sample)
        explainer = shap.KernelExplainer(predictor.predict_proba, x_samples)


        for i in tqdm(range(len(explain_data))):
            anomaly_shap_values = explainer.shap_values(explain_data[i])[explain_label[i]]
            f_imp.append(anomaly_shap_values)
        return np.array(f_imp)


