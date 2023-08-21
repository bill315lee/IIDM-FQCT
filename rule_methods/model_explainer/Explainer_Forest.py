import numpy as np


class Explainer_Forest:
    def __init__(self, sample_size, n_estimators=10):
        self.sample_size = sample_size
        self.n_trees = n_estimators

    def fit(self, data_normal, label_normal, explain_instance, explain_instance_label, improved=False):

        numX, numQ = data_normal.shape
        self.trees = []
        height_limit = 8
        for i in range(self.n_trees):
            idx = np.random.randint(numX, size=self.sample_size)
            X_sample = data_normal[idx, :]
            data_all = np.vstack((X_sample, explain_instance.reshape(1,-1)))
            label_all = np.array(list(label_normal[idx])+[explain_instance_label], dtype=int)
            tree = IsolationTree(height_limit)
            tree.fit(data_all, label_all, improved=improved)
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

    def fit(self, X, y, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        _, numQ = X.shape
        current_height = 0

        root = self.iTree(X,y, current_height, improved = improved)
        current_height = 1
        self.grow(root, current_height, improved = improved)
        self.root = root


        return self.root

    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)  # 数据集行数
        labelCounts = {}  # 声明保存每个标签（label）出现次数的字典
        for featVec in dataSet:  # 对每组特征向量进行统计
            currentLabel = featVec[-1]  # 提取标签信息
            if currentLabel not in labelCounts.keys():  # 如果标签没有放入统计次数的字典，添加进去
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1  # label计数
            # 以上是将每个标签出现的次数放入labelCounts字典中，目的就是求出香农公式里的P(x)
        shannonEnt = 0.0  # 经验熵
        for key in labelCounts:  # 计算经验熵
            prob = float(labelCounts[key]) / numEntries  # 选择该标签的概率
            shannonEnt -= prob * np.log(prob)  # 利用公式计算
        return shannonEnt  # 返回经验熵


    def SplitData(self, dataSet, axis, value):
        retDataSet = []  # 创建返回的数据集列表
        for featVec in dataSet:  # 遍历数据集
            if featVec[axis] == value:  # 如果
                reduceFeatVec = featVec[:axis]  # 去掉axis特征
                reduceFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
                retDataSet.append(reduceFeatVec)
        # 返回划分后的数据集
        return retDataSet


    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1  # 特征数量
        baseEntropy = self.calcShannonEnt(dataSet)  # 计数数据集的香农熵
        bestInfoGain = 0.0  # 信息增益
        bestFeature = -1  # 最优特征的索引值

        infoGain_lst = []
        for i in range(numFeatures):  # 循环的作用就是遍历所有特征
            featList = [example[i] for example in dataSet]  # 获取dataSet的第i个所有特征
            uniqueVals = set(featList)  # 创建set集合{}，元素不可重复
            newEntropy = 0.0  # 经验条件熵
            for value in uniqueVals:  # 计算信息增益
                subDataSet = self.SplitData(dataSet, i, value)  # subDataSet划分后的子集
                prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
                newEntropy += prob * self.calcShannonEnt((subDataSet))  # 根据公式计算经验条件熵
            infoGain = baseEntropy - newEntropy  # 信息增益
            # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
            infoGain_lst.append(infoGain)
        return infoGain_lst  # 最终返回的是信息增益最大特征的索引值



    def iTree(self, X, y, current_height, improved = False):



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

    def grow(self, node, current_height, improved = False):
        if type(node) == inNode:
            node.child = self.iTree(node.data, node.label, current_height, improved = improved)
            current_height += 1
            self.grow(node.child, current_height, improved = improved)

    def countNode(self, root):
        if type(root) == inNode:
            return 1 + self.countNode(root.left) + self.countNode(root.right)
        if type(root) == exNode:
            return 1
