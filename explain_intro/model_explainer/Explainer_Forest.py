import numpy as np


class Explainer_Forest:
    def __init__(self, sample_size, n_estimators=10):
        self.sample_size = sample_size
        self.n_trees = n_estimators

    def fit(self, data_normal, label_normal, explain_instance, explain_instance_label):

        numX, numQ = data_normal.shape
        self.trees = []
        height_limit = 8
        for i in range(self.n_trees):
            idx = np.random.randint(numX, size=self.sample_size)
            X_sample = data_normal[idx, :]
            data_all = np.vstack((X_sample, explain_instance.reshape(1,-1)))
            label_all = np.array(list(label_normal[idx])+[explain_instance_label], dtype=int)
            tree = IsolationTree(height_limit)
            tree.fit(data_all, label_all)
            self.trees.append(tree)

        return self



class inNode:
    def __init__(self, splitAtt, splitValue, inequality, data, label):
        self.data = data
        self.label = label
        self.inequality = inequality
        self.splitAtt = splitAtt
        self.splitValue = splitValue

class exNode:
    def __init__(self, size):
        self.size = size

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.inequality_lst = []
        self.feature_lst = []
        self.threshold_lst = []

    def fit(self, X, y):


        _, numQ = X.shape
        current_height = 0

        root = self.iTree(X,y, current_height)
        current_height = 1
        self.grow(root, current_height)
        self.root = root


        return self.root



    def iTree(self, X, y, current_height):



        if len(X) == 1 or current_height == self.height_limit:
            return exNode(X.shape[0])
        else:

            numX = len(X)

            attack_idx = np.where(y != 0)[0][0]
            normal_idx = np.where(y == 0)[0]

            normal_data = X[normal_idx]
            attack = X[attack_idx]

            C = normal_data - attack.reshape(1,-1)
            B = C.copy()
            B[B > 0] = 1
            B[B < 0] = 0

            A = C.copy()
            A[A > 0] = 0
            A[A < 0] = 1

            A_mean = np.mean(A, axis=0)
            B_mean = np.mean(B, axis=0)
            A_max = np.max(A_mean)
            B_max = np.max(B_mean)

            if A_max >= B_max:
                feature_idx = np.where(A_mean==A_max)[0]
                percentile_value_lst = []
                if len(feature_idx) > 1:
                    for idx in feature_idx:
                        Y = normal_data[:, idx] - attack[idx]
                        percentile_value_lst.append(np.percentile(np.abs(Y), 5))
                    feature_idx = feature_idx[np.argmax(percentile_value_lst)]
                else:
                    feature_idx = feature_idx[0]
                self.feature_lst.append(feature_idx)
                inequality = '>'
                self.inequality_lst.append(inequality)
                threshold = attack[feature_idx] - 0.5 * np.percentile(np.abs(normal_data[:, feature_idx] - attack[feature_idx]), 5)
                self.threshold_lst.append(threshold)
            else:
                feature_idx = np.where(B_mean==B_max)[0]
                percentile_value_lst = []
                if len(feature_idx) > 1:
                    for idx in feature_idx:
                        Y = normal_data[:, idx] - attack[idx]
                        percentile_value_lst.append(np.percentile(np.abs(Y), 5))
                    feature_idx = feature_idx[np.argmax(percentile_value_lst)]
                else:
                    feature_idx = feature_idx[0]
                self.feature_lst.append(feature_idx)
                inequality = '<'
                self.inequality_lst.append(inequality)
                threshold = attack[feature_idx] + 0.5 * np.percentile(np.abs(normal_data[:, feature_idx] - attack[feature_idx]), 5)
                self.threshold_lst.append(threshold)

            left_idx = set([i for i in range(numX) if X[i, feature_idx] < threshold])
            right_idx = set([i for i in range(numX)]) - left_idx

            if attack_idx in left_idx:
                pass_data = X[list(left_idx), :]
                pass_label = y[list(left_idx)]
            else:
                pass_data = X[list(right_idx), :]
                pass_label = y[list(right_idx)]

            return inNode(feature_idx, threshold, inequality, pass_data, pass_label)

    def grow(self, node, current_height):
        if type(node) == inNode:
            node.child = self.iTree(node.data, node.label, current_height)
            current_height += 1
            self.grow(node.child, current_height)

    def countNode(self, root):
        if type(root) == inNode:
            return 1 + self.countNode(root.left) + self.countNode(root.right)
        if type(root) == exNode:
            return 1
