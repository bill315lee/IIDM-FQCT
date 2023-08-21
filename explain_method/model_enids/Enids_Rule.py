import numpy as np
import pandas as pd
from utils import euclidian_dist

# class Enids_Forest:
#     def __init__(self, sample_size, n_estimators=10, percentile=5, tolerance=0.5):
#         self.sample_size = sample_size
#         self.n_trees = n_estimators
#         self.percentile = percentile
#         self.tolerance = tolerance
#
#
#     def fit(self, data_normal, label_normal, explain_instance, explain_instance_label, feature_idx):
#
#         numX = len(data_normal)
#         self.trees = []
#         height_limit = 8
#         for i in range(self.n_trees):
#             idx = np.random.randint(numX, size=self.sample_size)
#             X_sample = data_normal[idx, :]
#             data_all = np.vstack((X_sample, explain_instance.reshape(1,-1)))
#             label_all = np.array(list(label_normal[idx])+[explain_instance_label], dtype=int)
#             tree = IsolationTree(height_limit, self.percentile, self.tolerance)
#             tree.fit(data_all, label_all, feature_idx)
#             self.trees.append(tree)
#
#         return self
#
#
#
# class inNode:
#     def __init__(self, splitAtt, splitValue, inequality, data, label):
#         self.data = data
#         self.label = label
#         self.inequality = inequality
#         self.splitAtt = splitAtt
#         self.splitValue = splitValue
#
# class exNode:
#     def __init__(self, size):
#         self.size = size
#
# class IsolationTree:
#     def __init__(self, height_limit, percentile, tolerance):
#         self.height_limit = height_limit
#         self.inequality_lst = []
#         self.feature_lst = []
#         self.threshold_lst = []
#         self.percentile = percentile
#         self.tolerance = tolerance
#
#
#     def fit(self, X, y, feature_idx):
#
#
#         current_height = 0
#
#         root = self.iTree(X, y, current_height, feature_idx)
#         current_height = 1
#         self.grow(root, current_height, feature_idx)
#         self.root = root
#
#
#         return self.root
#
#     def calcShannonEnt(self, dataSet):
#         numEntries = len(dataSet)  # 数据集行数
#         labelCounts = {}  # 声明保存每个标签（label）出现次数的字典
#         for featVec in dataSet:  # 对每组特征向量进行统计
#             currentLabel = featVec[-1]  # 提取标签信息
#             if currentLabel not in labelCounts.keys():  # 如果标签没有放入统计次数的字典，添加进去
#                 labelCounts[currentLabel] = 0
#             labelCounts[currentLabel] += 1  # label计数
#             # 以上是将每个标签出现的次数放入labelCounts字典中，目的就是求出香农公式里的P(x)
#         shannonEnt = 0.0  # 经验熵
#         for key in labelCounts:  # 计算经验熵
#             prob = float(labelCounts[key]) / numEntries  # 选择该标签的概率
#             shannonEnt -= prob * np.log(prob)  # 利用公式计算
#         return shannonEnt  # 返回经验熵
#
#
#     def SplitData(self, dataSet, axis, value):
#         retDataSet = []  # 创建返回的数据集列表
#         for featVec in dataSet:  # 遍历数据集
#             if featVec[axis] == value:  # 如果
#                 reduceFeatVec = featVec[:axis]  # 去掉axis特征
#                 reduceFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
#                 retDataSet.append(reduceFeatVec)
#         # 返回划分后的数据集
#         return retDataSet
#
#
#     def chooseBestFeatureToSplit(self, dataSet):
#         numFeatures = len(dataSet[0]) - 1  # 特征数量
#
#         baseEntropy = self.calcShannonEnt(dataSet)  # 计数数据集的香农熵
#         bestInfoGain = 0.0  # 信息增益
#         bestFeature = -1  # 最优特征的索引值
#
#         infoGain_lst = []
#         for i in range(numFeatures):  # 循环的作用就是遍历所有特征
#             featList = [example[i] for example in dataSet]  # 获取dataSet的第i个所有特征
#             uniqueVals = set(featList)  # 创建set集合{}，元素不可重复
#             newEntropy = 0.0  # 经验条件熵
#             for value in uniqueVals:  # 计算信息增益
#                 subDataSet = self.SplitData(dataSet, i, value)  # subDataSet划分后的子集
#                 prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
#                 newEntropy += prob * self.calcShannonEnt((subDataSet))  # 根据公式计算经验条件熵
#             infoGain = baseEntropy - newEntropy  # 信息增益
#             # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
#             infoGain_lst.append(infoGain)
#         return infoGain_lst  # 最终返回的是信息增益最大特征的索引值
#
#
#
#
#     def iTree(self, X, y, current_height, feature_idx):
#
#
#         if len(X) == 1 or current_height == self.height_limit:
#             return exNode(X.shape[0])
#         else:
#             attack_idx = np.where(y != 0)[0][0]
#             normal_idx = np.where(y == 0)[0]
#
#             attack = X[attack_idx]
#             normal_data = X[normal_idx]
#             numX, numQ = X.shape
#             data_all = np.hstack((X, y.reshape(-1, 1)))
#             infoGain_lst = self.chooseBestFeatureToSplit(data_all.tolist())
#
#             infoGain_array = np.array(infoGain_lst)
#             max_value = np.max(infoGain_array)
#             max_idx = np.where(infoGain_array == max_value)[0]
#
#             if len(max_idx) == 1:
#                 splitAtt = max_idx[0]
#             else:
#                 percentile_value_lst = []
#                 for idx in max_idx:
#                     normdata_on_feaidx = data_all[normal_idx, idx]
#                     attack_on_feaidx = data_all[attack_idx, idx]
#                     value = np.abs(normdata_on_feaidx - attack_on_feaidx)
#                     percentile_value = np.percentile(value, self.percentile)
#                     percentile_value_lst.append(percentile_value)
#                 splitAtt = max_idx[np.argmax(percentile_value_lst)]
#
#             normal_max = np.max(normal_data[:, splitAtt])
#             normal_min = np.min(normal_data[:, splitAtt])
#             # three scenarios
#             attack_value = attack[splitAtt]
#             if attack_value < normal_min:
#                 splitValue = normal_min - 1e-2
#             elif attack_value > normal_max:
#                 splitValue = normal_max + 1e-2
#             else:
#                 Q_value = np.percentile(np.abs(normal_data[:, splitAtt] - attack_value), self.percentile)
#                 if np.median(normal_data[:, splitAtt]) <= attack_value:
#                     splitValue = attack_value - self.tolerance * Q_value
#                 else:
#                     splitValue = attack_value + self.tolerance * Q_value
#
#             left_idx = set([i for i in range(numX) if X[i, splitAtt] < splitValue])
#             right_idx = set([i for i in range(numX)]) - left_idx
#
#             if attack_idx in left_idx:
#                 inequality = '<'
#                 pass_data = X[list(left_idx), :]
#                 pass_label = y[list(left_idx)]
#             else:
#                 inequality = '>='
#                 pass_data = X[list(right_idx), :]
#                 pass_label = y[list(right_idx)]
#
#             self.inequality_lst.append(inequality)
#             self.feature_lst.append(feature_idx[splitAtt])
#             self.threshold_lst.append(splitValue)
#
#             return inNode(splitAtt, splitValue, inequality, pass_data, pass_label)
#
#
#
#     # def iTree(self, X, y, current_height, feature_idx):
#     #
#     #     if len(X) == 1 or current_height == self.height_limit:
#     #         return exNode(X.shape[0])
#     #     else:
#     #         attack_idx = np.where(y != 0)[0][0]
#     #         normal_idx = np.where(y == 0)[0]
#     #         numX, numQ = X.shape
#     #         data_all = np.hstack((X, y.reshape(-1,1)))
#     #         infoGain_lst = self.chooseBestFeatureToSplit(data_all.tolist())
#     #
#     #
#     #         infoGain_array = np.array(infoGain_lst)
#     #         max_value = np.max(infoGain_array)
#     #         max_idx = np.where(infoGain_array == max_value)[0]
#     #
#     #         if len(max_idx) == 1:
#     #             splitAtt = max_idx[0]
#     #         else:
#     #             percentile_value_lst = []
#     #             for idx in max_idx:
#     #                 normdata_on_feaidx = data_all[normal_idx, idx]
#     #                 attack_on_feaidx = data_all[attack_idx, idx]
#     #                 value = np.abs(normdata_on_feaidx - attack_on_feaidx)
#     #                 percentile_value = np.percentile(value, 5)
#     #                 percentile_value_lst.append(percentile_value)
#     #             splitAtt = max_idx[np.argmax(percentile_value_lst)]
#     #
#     #
#     #
#     #         normal_value = X[normal_idx, splitAtt]
#     #         normal_value_mean = np.mean(normal_value)
#     #
#     #         anomaly_value = X[attack_idx, splitAtt]
#     #
#     #         Q_value = np.percentile(np.abs(normal_value - anomaly_value), 5)
#     #         if normal_value_mean <= anomaly_value:
#     #             splitValue = anomaly_value - 0.5 * Q_value
#     #         else:
#     #             splitValue = anomaly_value + 0.5 * Q_value
#     #
#     #
#     #         left_idx = set([i for i in range(numX) if X[i, splitAtt] < splitValue])
#     #         right_idx = set([i for i in range(numX)]) - left_idx
#     #
#     #         if attack_idx in left_idx:
#     #             inequality = '<'
#     #             pass_data = X[list(left_idx), :]
#     #             pass_label = y[list(left_idx)]
#     #         else:
#     #             inequality = '>='
#     #             pass_data = X[list(right_idx), :]
#     #             pass_label = y[list(right_idx)]
#     #
#     #         self.inequality_lst.append(inequality)
#     #         self.feature_lst.append(feature_idx[splitAtt])
#     #         self.threshold_lst.append(splitValue)
#     #
#     #         return inNode(splitAtt, splitValue, inequality, pass_data, pass_label)
#
#     def grow(self, node, current_height, feature_idx):
#         if type(node) == inNode:
#             node.child = self.iTree(node.data, node.label, current_height, feature_idx)
#             current_height += 1
#             self.grow(node.child, current_height, feature_idx)
#
#     def countNode(self, root):
#         if type(root) == inNode:
#             return 1 + self.countNode(root.left) + self.countNode(root.right)
#         if type(root) == exNode:
#             return 1

# class Enids_Forest:
#     def __init__(self, sample_size, n_estimators=10, percentile=5, tolerance=0.5, eps=1e-6):
#         self.sample_size = sample_size
#         self.n_trees = n_estimators
#         self.percentile = percentile
#         self.tolerance = tolerance
#         self.eps = eps
#
#     def fit(self, data_normal, label_normal, explain_instance, explain_instance_label, feature_idx):
#
#         numX = len(data_normal)
#         self.trees = []
#         height_limit = 8
#         for i in range(self.n_trees):
#             idx = np.random.randint(numX, size=self.sample_size)
#             X_sample = data_normal[idx, :]
#             data_all = np.vstack((X_sample, explain_instance.reshape(1,-1)))
#             label_all = np.array(list(label_normal[idx])+[explain_instance_label], dtype=int)
#             tree = IsolationTree(height_limit, self.percentile, self.tolerance, self.eps)
#             tree.fit(data_all, label_all, feature_idx, self.eps)
#             self.trees.append(tree)
#
#         return self
#
#
#
# class inNode:
#     def __init__(self, splitAtt, splitValue, inequality, data, label):
#         self.data = data
#         self.label = label
#         self.inequality = inequality
#         self.splitAtt = splitAtt
#         self.splitValue = splitValue
#
# class exNode:
#     def __init__(self, size):
#         self.size = size
#
# class IsolationTree:
#     def __init__(self, height_limit, percentile, tolerance, eps):
#         self.height_limit = height_limit
#         self.inequality_lst = []
#         self.feature_lst = []
#         self.threshold_lst = []
#         self.percentile = percentile
#         self.tolerance = tolerance
#         self.eps = eps
#
#     def fit(self, X, y, feature_idx, eps):
#
#
#         current_height = 0
#
#         root = self.iTree(X, y, current_height, feature_idx)
#         current_height = 1
#         self.grow(root, current_height, feature_idx)
#         self.root = root
#
#
#         return self.root
#
#     def calcShannonEnt(self, dataSet):
#         numEntries = len(dataSet)  # 数据集行数
#         labelCounts = {}  # 声明保存每个标签（label）出现次数的字典
#         for featVec in dataSet:  # 对每组特征向量进行统计
#             currentLabel = featVec[-1]  # 提取标签信息
#             if currentLabel not in labelCounts.keys():  # 如果标签没有放入统计次数的字典，添加进去
#                 labelCounts[currentLabel] = 0
#             labelCounts[currentLabel] += 1  # label计数
#             # 以上是将每个标签出现的次数放入labelCounts字典中，目的就是求出香农公式里的P(x)
#         shannonEnt = 0.0  # 经验熵
#         for key in labelCounts:  # 计算经验熵
#             prob = float(labelCounts[key]) / numEntries  # 选择该标签的概率
#             shannonEnt -= prob * np.log(prob)  # 利用公式计算
#         return shannonEnt  # 返回经验熵
#
#
#     def SplitData(self, dataSet, axis, value):
#         retDataSet = []  # 创建返回的数据集列表
#         for featVec in dataSet:  # 遍历数据集
#             if featVec[axis] == value:  # 如果
#                 reduceFeatVec = featVec[:axis]  # 去掉axis特征
#                 reduceFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
#                 retDataSet.append(reduceFeatVec)
#         # 返回划分后的数据集
#         return retDataSet
#
#
#     def chooseBestFeatureToSplit(self, dataSet):
#         numFeatures = len(dataSet[0]) - 1  # 特征数量
#
#         baseEntropy = self.calcShannonEnt(dataSet)  # 计数数据集的香农熵
#         bestInfoGain = 0.0  # 信息增益
#         bestFeature = -1  # 最优特征的索引值
#
#         infoGain_lst = []
#         for i in range(numFeatures):  # 循环的作用就是遍历所有特征
#             featList = [example[i] for example in dataSet]  # 获取dataSet的第i个所有特征
#             uniqueVals = set(featList)  # 创建set集合{}，元素不可重复
#             newEntropy = 0.0  # 经验条件熵
#             for value in uniqueVals:  # 计算信息增益
#                 subDataSet = self.SplitData(dataSet, i, value)  # subDataSet划分后的子集
#                 prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
#                 newEntropy += prob * self.calcShannonEnt((subDataSet))  # 根据公式计算经验条件熵
#             infoGain = baseEntropy - newEntropy  # 信息增益
#             # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
#             infoGain_lst.append(infoGain)
#         return infoGain_lst  # 最终返回的是信息增益最大特征的索引值
#
#
#     # def determine_feature_threshold(self, normal_data, attack, numQ, percentile, tolerance):
#     #
#     #     # 第1个判断，判断是否远离所有正常
#     #     feature_tmp = []
#     #     threshold_tmp = []
#     #     dist = []
#     #     for k in range(numQ):
#     #         normal_values = normal_data[:,k]
#     #         attack_value = attack[k]
#     #         if attack_value + self.eps > np.max(normal_values):
#     #             feature_tmp.append(k)
#     #             threshold_tmp.append(np.max(normal_values) + 0.001)
#     #             dist.append(np.abs(np.max(normal_values) - attack_value))
#     #         elif attack_value < np.min(normal_values) + self.eps:
#     #             feature_tmp.append(k)
#     #             threshold_tmp.append(np.min(normal_values) - 0.001)
#     #             dist.append(np.abs(np.min(normal_values) - attack_value))
#     #         else:
#     #             pass
#     #
#     #     if len(feature_tmp) == 1:
#     #         return feature_tmp[0],  threshold_tmp[0]
#     #
#     #     elif len(feature_tmp) > 1:
#     #         most_idx = np.argmax(dist)
#     #         return feature_tmp[most_idx],  threshold_tmp[most_idx]
#     #     else:
#     #         # 第2个判断，判断是否在分为数内
#     #         feature_tmp = []
#     #         threshold_tmp = []
#     #         dist = []
#     #         for k in range(numQ):
#     #             normal_values = normal_data[:, k]
#     #             attack_value = attack[k]
#     #             upper_percent = np.percentile(normal_values, 100 - percentile)
#     #             lower_percent = np.percentile(normal_values, percentile)
#     #
#     #             if attack_value + self.eps >= upper_percent:
#     #                 feature_tmp.append(k)
#     #                 threshold_tmp.append(upper_percent)
#     #                 dist.append(np.abs(attack_value - upper_percent))
#     #
#     #             elif attack_value <= lower_percent + self.eps:
#     #                 feature_tmp.append(k)
#     #                 threshold_tmp.append(lower_percent)
#     #                 dist.append(np.abs(np.min(normal_values) - attack_value))
#     #             else:
#     #                 pass
#     #
#     #         if len(feature_tmp) == 1:
#     #             return feature_tmp[0], threshold_tmp[0]
#     #
#     #         elif len(feature_tmp) > 1:
#     #             most_idx = np.argmax(dist)
#     #             return feature_tmp[most_idx],  threshold_tmp[most_idx]
#     #
#     #         else:
#     #             # 第3个判断，都落在了正常数据内
#     #             feature_tmp = []
#     #             threshold_tmp = []
#     #             dist = []
#     #             for k in range(numQ):
#     #                 normal_values = normal_data[:, k]
#     #                 attack_value = attack[k]
#     #                 Q_value = np.percentile(np.abs(normal_values - attack_value), percentile)
#     #                 if np.median(normal_values) <= attack_value + self.eps:
#     #                     feature_tmp.append(k)
#     #                     threshold = attack_value - tolerance * Q_value
#     #                     threshold_tmp.append(threshold)
#     #                     dist.append(tolerance * Q_value)
#     #                 else:
#     #                     feature_tmp.append(k)
#     #                     threshold = attack_value + tolerance * Q_value
#     #                     threshold_tmp.append(threshold)
#     #                     dist.append(tolerance * Q_value)
#     #             if len(feature_tmp) == 1:
#     #                 return feature_tmp[0],  threshold_tmp[0]
#     #
#     #             elif len(feature_tmp) > 1:
#     #                 most_idx = np.argmax(dist)
#     #                 return feature_tmp[most_idx],  threshold_tmp[most_idx]
#
#
#
#     def iTree(self, X, y, current_height, feature_idx):
#
#         if len(X) == 1 or current_height == self.height_limit:
#             return exNode(X.shape[0])
#         else:
#             attack_idx = np.where(y != 0)[0][0]
#             normal_idx = np.where(y == 0)[0]
#             numX, numQ = X.shape
#             data_all = np.hstack((X, y.reshape(-1,1)))
#             infoGain_lst = self.chooseBestFeatureToSplit(data_all.tolist())
#
#
#             infoGain_array = np.array(infoGain_lst)
#             max_value = np.max(infoGain_array)
#             max_idx = np.where(infoGain_array == max_value)[0]
#
#             if len(max_idx) == 1:
#                 splitAtt = max_idx[0]
#             else:
#                 percentile_value_lst = []
#                 for idx in max_idx:
#                     normdata_on_feaidx = data_all[normal_idx, idx]
#                     attack_on_feaidx = data_all[attack_idx, idx]
#                     value = np.abs(normdata_on_feaidx - attack_on_feaidx)
#                     percentile_value = np.percentile(value, 5)
#                     percentile_value_lst.append(percentile_value)
#                 splitAtt = max_idx[np.argmax(percentile_value_lst)]
#
#
#
#             normal_value = X[normal_idx, splitAtt]
#             normal_value_mean = np.mean(normal_value)
#
#             anomaly_value = X[attack_idx, splitAtt]
#
#             Q_value = np.percentile(np.abs(normal_value - anomaly_value), 5)
#             if normal_value_mean <= anomaly_value:
#                 splitValue = anomaly_value - 0.5 * Q_value
#             else:
#                 splitValue = anomaly_value + 0.5 * Q_value
#
#
#             left_idx = set([i for i in range(numX) if X[i, splitAtt] < splitValue])
#             right_idx = set([i for i in range(numX)]) - left_idx
#
#             if attack_idx in left_idx:
#                 inequality = '<'
#                 pass_data = X[list(left_idx), :]
#                 pass_label = y[list(left_idx)]
#             else:
#                 inequality = '>='
#                 pass_data = X[list(right_idx), :]
#                 pass_label = y[list(right_idx)]
#
#             self.inequality_lst.append(inequality)
#             self.feature_lst.append(feature_idx[splitAtt])
#             self.threshold_lst.append(splitValue)
#
#             return inNode(splitAtt, splitValue, inequality, pass_data, pass_label)
#
#     def grow(self, node, current_height, feature_idx):
#         if type(node) == inNode:
#             node.child = self.iTree(node.data, node.label, current_height, feature_idx)
#             current_height += 1
#             self.grow(node.child, current_height, feature_idx)
#
#     def countNode(self, root):
#         if type(root) == inNode:
#             return 1 + self.countNode(root.left) + self.countNode(root.right)
#         if type(root) == exNode:
#             return 1



class Enids_Forest:
    def __init__(self, sample_size, n_estimators=10, percentile=5, tolerance=0.5, eps= 1e-6):
        self.sample_size = sample_size
        self.n_trees = n_estimators
        self.percentile = percentile
        self.tolerance = tolerance
        self.eps = eps

    def fit(self, data_normal, label_normal, explain_instance, explain_instance_label, feature_idx, size_array):

        numX = len(data_normal)
        self.trees = []
        height_limit = 8
        for i in range(self.n_trees):
            idx = np.random.randint(numX, size=self.sample_size)
            X_sample = data_normal[idx, :]
            size_array = size_array[idx]
            data_all = np.vstack((X_sample, explain_instance.reshape(1,-1)))
            label_all = np.array(list(label_normal[idx])+[explain_instance_label], dtype=int)
            size_all = np.array(list(size_array)+[1], dtype=int)
            tree = IsolationTree(height_limit, self.percentile, self.tolerance, self.eps)
            tree.fit(data_all, label_all, feature_idx, size_all)
            self.trees.append(tree)

        return self



class inNode:
    def __init__(self, splitAtt, splitValue, inequality, data, label, proto_size):
        self.data = data
        self.label = label
        self.inequality = inequality
        self.splitAtt = splitAtt
        self.splitValue = splitValue
        self.proto_size = proto_size

class exNode:
    def __init__(self, size):
        self.size = size

class IsolationTree:
    def __init__(self, height_limit, percentile, tolerance, eps):
        self.height_limit = height_limit
        self.inequality_lst = []
        self.feature_lst = []
        self.threshold_lst = []
        self.percentile = percentile
        self.tolerance = tolerance
        self.eps = eps

    def fit(self, X, y, feature_idx, size_all):

        self.stop_threshold = np.max([int(self.percentile * 0.01 * np.sum(size_all)), 1])

        current_height = 0

        root = self.iTree(X, y, current_height, feature_idx, size_all)
        current_height = 1
        self.grow(root, current_height, feature_idx)
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


    def determine_threshold(self, normal_values, attack_value):

        # 第1个判断，判断是否远离所有正常
        if attack_value + self.eps > np.max(normal_values):
            threshold = np.max(normal_values) + 0.001
        elif attack_value < np.min(normal_values) + self.eps:
            threshold = np.min(normal_values) - 0.001

        else:
            upper_percent = np.percentile(normal_values, 100 - self.percentile)
            lower_percent = np.percentile(normal_values, self.percentile)

            if attack_value + self.eps >= upper_percent:
                threshold = upper_percent

            elif attack_value <= lower_percent + self.eps:
                threshold = lower_percent
            else:
                # 第3个判断，都落在了正常数据内
                Q_value = np.percentile(np.abs(normal_values - attack_value), self.percentile)
                if np.median(normal_values) <= attack_value + self.eps:
                    threshold = attack_value - self.tolerance * Q_value
                else:
                    threshold = attack_value + self.tolerance * Q_value

        return threshold

    def iTree(self, X, y, current_height, feature_idx, size_all):

        # if len(X) == 1 or current_height == self.height_limit:
        #     return exNode(X.shape[0])
        if np.sum(size_all) <= self.stop_threshold or current_height == self.height_limit:
            return exNode(X.shape[0])
        else:
            attack_idx = np.where(y != 0)[0][0]
            normal_idx = np.where(y == 0)[0]

            numX, numQ = X.shape
            data_all = np.hstack((X, y.reshape(-1,1)))
            infoGain_lst = self.chooseBestFeatureToSplit(data_all.tolist())


            infoGain_array = np.array(infoGain_lst)
            max_value = np.max(infoGain_array)
            max_idx = np.where(infoGain_array == max_value)[0]

            if len(max_idx) == 1:
                splitAtt = max_idx[0]
            else:
                percentile_value_lst = []
                for idx in max_idx:
                    normdata_on_feaidx = data_all[normal_idx, idx]
                    # normal_values_weights = []
                    # size_normal = size_all[normal_idx]
                    # for j in range(len(normdata_on_feaidx)):
                    #     for _ in range(size_normal[j]):
                    #         normal_values_weights.append(normdata_on_feaidx[j])
                    # normal_values_weights_array = np.array(normal_values_weights)
                    attack_on_feaidx = data_all[attack_idx, idx]
                    value = np.abs(normdata_on_feaidx - attack_on_feaidx)
                    percentile_value = np.percentile(value, self.percentile)
                    percentile_value_lst.append(percentile_value)
                splitAtt = max_idx[np.argmax(percentile_value_lst)]

            # normal_values = X[normal_idx, splitAtt]
            # attack_value = X[attack_idx, splitAtt]
            # splitValue = self.determine_threshold(normal_values, attack_value)
            normal_value = X[normal_idx, splitAtt]
            # normal_values_weights = []
            # size_normal = size_all[normal_idx]
            # for j in range(len(normal_value)):
            #     for _ in range(size_normal[j]):
            #         normal_values_weights.append(normal_value[j])
            # normal_values_weights_array = np.array(normal_values_weights)

            normal_value_median = np.median(normal_value)
            anomaly_value = X[attack_idx, splitAtt]
            Q_value = np.percentile(np.abs(normal_value - anomaly_value), self.percentile)
            if normal_value_median <= anomaly_value:
                splitValue = anomaly_value - self.tolerance * Q_value
            else:
                splitValue = anomaly_value + self.tolerance * Q_value


            left_idx = set([i for i in range(numX) if X[i, splitAtt] < splitValue])
            right_idx = set([i for i in range(numX)]) - left_idx

            if attack_idx in left_idx:
                inequality = '<'
                pass_data = X[list(left_idx), :]
                pass_label = y[list(left_idx)]
                pass_size = size_all[list(left_idx)]
            else:
                inequality = '>='
                pass_data = X[list(right_idx), :]
                pass_label = y[list(right_idx)]
                pass_size = size_all[list(right_idx)]


            self.inequality_lst.append(inequality)
            self.feature_lst.append(feature_idx[splitAtt])
            self.threshold_lst.append(splitValue)

            return inNode(splitAtt, splitValue, inequality, pass_data, pass_label, pass_size)

    def grow(self, node, current_height, feature_idx):
        if type(node) == inNode:
            node.child = self.iTree(node.data, node.label, current_height, feature_idx, node.proto_size)
            current_height += 1
            self.grow(node.child, current_height, feature_idx)

    def countNode(self, root):
        if type(root) == inNode:
            return 1 + self.countNode(root.left) + self.countNode(root.right)
        if type(root) == exNode:
            return 1

def Enids_rule_generation(feature_lst_explain, LGP_idx, data_test, label_test, proto_array, eps):


    loss_guide_use = True
    keypoints_use = True
    percentile = 5
    tolerance = 0.5

    center_array = np.array([proto.center for proto in proto_array])
    r_array = np.array([proto.r for proto in proto_array])
    size_array = np.array([proto.size for proto in proto_array])

    attack_dict = {}

    feature_lst_alldata = []
    inequality_lst_alldata = []
    threshold_lst_alldata = []
    rule_num = []

    rule_dict = {}

    for i in range(len(data_test)):
        print('i=', i)
        explain_instance = data_test[i]
        explain_label = label_test[i]
        if loss_guide_use:
            new_center_array = center_array[LGP_idx[i]]
            new_r_array = r_array[LGP_idx[i]]
            new_size_array = size_array[LGP_idx[i]]
        else:
            new_center_array = center_array
            new_r_array = r_array
            new_size_array = size_array

        if explain_label in attack_dict:
            attack_dict[explain_label].append(explain_instance)
        else:
            attack_dict[explain_label] = [explain_instance]
        attack_mean = np.mean(np.array(attack_dict[explain_label]), axis=0)

        if keypoints_use:
            d = euclidian_dist(explain_instance, new_center_array)
            keypoints = ((d - new_r_array) / d).reshape(-1, 1) * (
                    new_center_array - attack_mean.reshape(1, -1)) + attack_mean.reshape(1, -1)
        else:
            keypoints = new_center_array


        feature_idx = np.array(feature_lst_explain[i])

        en_forest = Enids_Forest(sample_size=len(new_center_array), n_estimators=1, percentile=percentile, tolerance=tolerance, eps=eps)

        en_forest.fit(keypoints[:, feature_idx], np.zeros(len(keypoints)),
                      explain_instance[feature_idx].reshape(1, -1), 1, feature_idx, new_size_array)

        # inequality_lst = en_forest.trees[0].inequality_lst
        # feature_lst = en_forest.trees[0].feature_lst
        # threshold_lst = en_forest.trees[0].threshold_lst

        H_subspace = {}
        for t in en_forest.trees:
            inequality_lst = t.inequality_lst
            feature_lst = t.feature_lst
            threshold_lst = t.threshold_lst

            for j in range(len(inequality_lst)):
                inequality = inequality_lst[j]
                feature = feature_lst[j]
                threshold = threshold_lst[j]
                tag = str(feature) + '_' + inequality
                if tag in H_subspace:
                    H_subspace[tag].append(threshold)
                else:
                    H_subspace[tag] = [threshold]

        rule_tags = np.array(list(H_subspace.keys()))
        rule_thresholds = np.array(list(H_subspace.values()))
        rule_nums = np.array([len(rule) for rule in rule_thresholds])
        all_rule_nums = np.sum(rule_nums)

        sum_k_rules = 0
        idx_rules = []
        for idx in np.argsort(-rule_nums):
            sum_k_rules += rule_nums[idx]
            idx_rules.append(idx)
            if sum_k_rules / all_rule_nums > 0.9:
                break
        rule_choose = rule_tags[idx_rules]
        rule_thresholds_choose = rule_thresholds[idx_rules]

        feature_lst = []
        inequality_lst = []
        threshold_lst = []

        for j in range(len(rule_choose)):
            tag = rule_choose[j]
            feature_lst.append(int(tag.split('_')[0]))
            inequality_lst.append(tag.split('_')[1])
            threshold_lst.append(np.median(rule_thresholds_choose[j]))

        feature_lst_alldata.append(feature_lst)
        inequality_lst_alldata.append(inequality_lst)
        threshold_lst_alldata.append(threshold_lst)
        rule_num.append(len(set(feature_lst)))

        #     # incremental rule_dict
        #     if explain_label not in rule_dict:
        #         rule_lst = []
        #         for j in range(len(feature_lst)):
        #             f_i_t = str(feature_lst[j])+'_'+inequality_lst[j]+'_'+str(threshold_lst[j])
        #             rule_lst.append(f_i_t)
        #         rule_dict[explain_label] = rule_lst
        #     else:
        #         rules = rule_dict[explain_label]
        #         print('now=')
        #         print(feature_lst)
        #         print(inequality_lst)
        #         print(threshold_lst)
        #         print('history=')
        #         print(rules)
        #         exit()
        #
        #
        #
        # wrong_normal = 0
        # for keypoint in keypoints:
        #     outlier_num = 0
        #     for j in range(len(feature_lst)):
        #         if inequality_lst[j] == '>=' and keypoint[feature_lst[j]] + eps >= threshold_lst[j]:
        #             outlier_num = +1
        #         elif inequality_lst[j] == '<' and keypoint[feature_lst[j]] < threshold_lst[j] + eps:
        #             outlier_num = +1
        #         else:
        #             pass
        #     if outlier_num == len(feature_lst):
        #         wrong_normal += 1
        #         print(keypoint, explain_instance, threshold_lst[0])
        # print('wrong_normal=', wrong_normal)




    return feature_lst_alldata, inequality_lst_alldata, threshold_lst_alldata