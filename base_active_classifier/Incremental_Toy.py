import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import copy
from heapq import *
import pickle

class Incremental_Prototype_Toy(object):
    class Prototype:
        # init
        def __init__(self, feature, label, idx, size):
            if isinstance(idx, list):
                self.center = feature
                self.data_idx_lst = idx
                self.recent_idx = idx[-1]
                # 这时传进来的数据是个lst
                # self.data = feature
                self.size = size
                self.ss = 0
                self.ls = 0
                self.r = 0
                self.label = label

            else:
                self.center = feature
                self.data_idx_lst = [idx]
                self.recent_idx = idx
                self.size = size
                self.ls = feature
                self.ss = feature ** 2
                self.r = 0
                self.instances = []
                self.instances.append(feature)

                self.label = label

        def update_proto(self, feature, label, idx):

            self.center = (self.center * self.size + feature) / (self.size + 1)
            self.size += 1
            self.data_idx_lst.append(idx)
            self.recent_idx = np.max([self.recent_idx, idx])
            self.ls = self.ls + feature
            self.ss = self.ss + feature ** 2
            self.instances.append(feature)

            # item = np.sum(self.size * self.ss - self.ls**2) / self.size**2
            # self.r = np.sqrt(item + 1e-6)
            instances_array = np.array(self.instances)
            # if len(instances_array) > 1000:
            #     sample_idx = np.random.choice(np.arange(len(instances_array)), 1000, replace=False)
            #     instances_array = instances_array[sample_idx]
            self.r = np.max(self.euclidian_dist(instances_array, self.center))


        @staticmethod
        def euclidian_dist(a, b):
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def __init__(self, threshold):
        super().__init__()
        self._dict = {}
        self.threshold = threshold

    def assign(self, feature, label, index):

        if label not in self._dict:
            prototype = Incremental_Prototype_Toy.Prototype(feature, label, index, 1)
            self._dict[label] = [prototype]
        else:
            center_array = np.array([proto.center for proto in self._dict[label]])
            distance = self.euclidian_dist(feature, center_array)
            min_dist_idx = np.argmin(distance)
            min_dist = distance[min_dist_idx]

            if min_dist <= self.threshold:
                proto = self._dict[label][min_dist_idx]
                proto.update_proto(feature, label, index)
            else:
                prototype = Incremental_Prototype_Toy.Prototype(feature, label, index, 1)

                if label in self._dict:
                    self._dict[label].append(prototype)
                else:
                    self._dict[label] = [prototype]


    @staticmethod
    def euclidian_dist(a, b):
        if a.ndim == 1:
            a = a.reshape(1,-1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))


    def merge(self, label):

        proto_lst_label = self._dict[label]
        proto_lst = pickle.loads(pickle.dumps(proto_lst_label))

        centers = np.array([proto.center for proto in proto_lst])
        r_array = np.array([proto.r for proto in proto_lst])

        dist_paire = []
        dist_matrix = pairwise_distances(centers)
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                heappush(dist_paire, (dist_matrix[i,j], (i,j), (r_array[i], r_array[j])))

        tag_index = []

        # while dist_paire[0][0] < self.threshold and len(dist_paire) != 1:
        while dist_paire[0][0] <= np.max([dist_paire[0][2][0], dist_paire[0][2][1]]) and len(dist_paire)!=1:
        # while dist_paire[0][0] < dist_paire[0][2][0] + dist_paire[0][2][1] and len(dist_paire) != 1:

            if dist_paire[0][1][0] in tag_index or dist_paire[0][1][1] in tag_index:
                heappop(dist_paire)
                continue
            else:
                index_a = dist_paire[0][1][0]
                index_b = dist_paire[0][1][1]
                size = proto_lst[index_a].size + proto_lst[index_b].size
                heappop(dist_paire)
                tag_index.append(index_a)
                tag_index.append(index_b)
                center = (proto_lst[index_a].center * proto_lst[index_a].size + proto_lst[index_b].center * proto_lst[index_b].size) / size
                data_idx_lst = proto_lst[index_a].data_idx_lst + proto_lst[index_b].data_idx_lst
                sort_idx = np.argsort(np.array(data_idx_lst))
                proto = Incremental_Prototype_Toy.Prototype(center, label, list(np.array(data_idx_lst)[sort_idx]), size)
                proto.ls = proto_lst[index_a].ls+proto_lst[index_b].ls
                proto.ss = proto_lst[index_a].ss+proto_lst[index_b].ss
                instances_lst = proto_lst[index_a].instances+proto_lst[index_b].instances
                proto.instances = list(np.array(instances_lst)[sort_idx])

                # item = np.sum(proto.size * proto.ss - proto.ls ** 2) / proto.size ** 2
                # proto.r = np.sqrt(item + 1e-6)
                instances_array = np.array(proto.instances)
                if len(instances_array) > 1000:
                    sample_idx = np.random.choice(np.arange(len(instances_array)), 1000, replace=False)
                    instances_array = instances_array[sample_idx]
                proto.r = np.max(self.euclidian_dist(instances_array, proto.center))

                proto_lst.append(proto)

                for i in range(len(proto_lst)):
                    if (i not in tag_index) and (i != len(proto_lst)-1):
                        heappush(dist_paire, (self.euclidian_dist(proto_lst[i].center, proto.center), (i, len(proto_lst)-1), (proto_lst[i].r, proto.r)))

        self._dict[label] = []
        for i in range(len(proto_lst)):
            if i not in tag_index:
                self._dict[label].append(proto_lst[i])