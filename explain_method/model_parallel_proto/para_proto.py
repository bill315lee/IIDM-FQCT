
from explain_method.model_parallel_proto.Incremental_Prototypes import Incremental_Prototype
import numpy as np
from tqdm import tqdm
# from explain_method.model_parallel_proto.Optimization import Optimization
from explain_method.model_parallel_proto.Optimization import Optimization_experimetns
from utils import SD_diff, num_to_subspace, euclidian_dist

class Para_Proto:

    def __init__(self, args, X_train):
        self.ipdt = args.ipdt
        self.datatype = args.datatype
        self.train_date = args.train_date
        self.device = args.device

        self.MIN_VALUE = np.min(X_train, axis=0)
        self.MAX_VALUE = np.max(X_train, axis=0)
        self.dim = X_train.shape[1]
        # self.dim_length = np.sqrt(self.grid_size * self.grid_size / self.dim)
        # self.cov_matrix = np.cov(X_train.T)

        self.exp_method = args.exp_method
        self.device = args.device
        self.train_size = len(X_train)

    def proto_fit(self, data, label):
        train_size = len(data)

        protos = Incremental_Prototype(self.ipdt)
        for i in tqdm(range(len(data))):
            protos.assign(data[i], label[i], i - train_size)

        before_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        for label_key in set(label):
            protos.merge(label_key)
        after_merge_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        print(f'Before Merge={before_len}, After Merge={after_merge_len}')

        return protos._dict


    def proto_update(self, data, label, proto_dict):

        protos = Incremental_Prototype(self.ipdt)
        protos._dict = proto_dict
        for i in range(len(data)):
            protos.assign(data[i], label[i], i - self.train_size)
        before_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        for label_key in set(label):
            protos.merge(label_key)
        after_merge_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        proto_dict =  protos._dict
        # proto_dict = {}
        # for label_key in  protos._dict:
        #     proto_lst = []
        #     for proto in protos._dict[label_key]:
        #         if proto.size > 1:
        #             proto_lst.append(proto)
        #     proto_dict[label_key] = proto_lst
        #
        # after_delete_len = np.sum([len(proto_dict[label_key]) for label_key in proto_dict])
        #
        print(f'Before Merge={before_len}, After Merge={after_merge_len}')

        return proto_dict

    def explain(self, explain_data, explain_label, proto_dict):
        explanation_lst = []
        opt = Optimization_experimetns(epochs=500, lr=1, dim=self.dim, device=self.device, nbr_num=5)
        for i in range(len(explain_data)):
            # # 解释数据
            explanation = opt.explain(explain_data[i], explain_label[i], proto_dict)
            explanation_lst.append(explanation)


            # 更新解释模型
            # protos = Incremental_Prototype(self.ipdt)
            # protos._dict = proto_dict
            # protos.assign(explain_data[i], explain_label[i], i - self.train_size)
            # if len(protos._dict[explain_label[i]]) > 1:
            #     protos.merge(explain_label[i])
            # proto_dict = protos._dict


        return explanation_lst


#     # 先判断是否属于现有特征集合
        #     ToExplain_tag = False
        #     if explain_label[i] in subproto_dict:
        #
        #         key_cover_dict = {}
        #         key_subproto_idx_dict = {}
        #
        #         for key in subproto_dict[explain_label[i]]:
        #             subprotos = subproto_dict[explain_label[i]][key]
        #             centers = np.array([proto.center for proto in subprotos])
        #             rs = np.array([proto.r for proto in subprotos])
        #
        #             subspace = num_to_subspace(key)
        #             explain_instance_subspace = explain_data[i][subspace]
        #             dist = euclidian_dist(explain_instance_subspace, centers)
        #             cover_idx = np.where(dist <= rs)[0]
        #
        #             if len(cover_idx) != 0:
        #                 key_cover_dict[key] = 1
        #                 ratio = dist[cover_idx] / rs[cover_idx]
        #                 subproto_idx = cover_idx[np.argmin(ratio)]
        #                 key_subproto_idx_dict[key] = (subproto_idx, np.min(ratio))
        #
        #             else:
        #                 key_cover_dict[key] = 0
        #
        #         key_cover_value = np.array(list(key_cover_dict.values()))
        #         key_cover_key = np.array(list(key_cover_dict.keys()))
        #
        #         # 没有落入任何子空间的任何原型中
        #         if (key_cover_value == 0).all():
        #             ToExplain_tag = True
        #         # 有落入某个子空间的某个原型中，更新
        #         else:
        #             # 找到落入的那些子空间
        #             key = key_cover_key[np.where(key_cover_value == 1)[0]]
        #             if len(key) != 1:
        #
        #                 key_subproto_idx_values = np.array(list(key_subproto_idx_dict.values()))
        #                 key_subproto_idx_key = np.array(list(key_subproto_idx_dict.keys()))
        #
        #                 key = key_subproto_idx_key[np.argmin([tuple[1] for tuple in key_subproto_idx_values])]
        #             else:
        #                 key = key[0]
        #             subspace = num_to_subspace(key)
        #             subproto = Incremental_Prototype(self.ipdt * np.sqrt(len(subspace)/self.dim))
        #             subproto._dict[explain_label[i]] = subproto_dict[explain_label[i]][key]
        #
        #
        #             subproto.assign(explain_data[i][subspace], explain_label[i], i - self.train_size)
        #             subproto_dict[explain_label[i]][key] = subproto._dict[explain_label[i]]
        #             explanation = np.zeros(self.dim)
        #             explanation[subspace] = 1
        #     # 不属于现有特征集合
        #     else:
        #         ToExplain_tag = True
        #
        #
        #     if ToExplain_tag == True:
        #         explanation = opt.explain(explain_data[i], explain_label[i], proto_dict)
        #         subspace = SD_diff(explanation, nbr_num=3)
        #         protos = Incremental_Prototype(self.ipdt * np.sqrt(len(subspace)/self.dim))
        #         protos._dict = proto_dict
        #         protos.assign(explain_data[i], explain_label[i], i - self.train_size)
        #         proto_dict = protos._dict
        #         # print('proto_size_for_label:', [len(proto_dict[label_key]) for label_key in proto_dict])
        #         # print(SD_diff(explanation, nbr_num=3))
        #
        #
        #         subspace_key = int(np.sum(10 ** subspace))
        #
        #         if explain_label[i] in subproto_dict:
        #             if subspace_key in subproto_dict[explain_label[i]]:
        #                 # 存在标签，存在对应的特征集合（其下有很多proto，之前肯定是查看过，他没有落入任何一个圆中，所以对这些圆进行一个更新）
        #                 subproto = Incremental_Prototype(self.ipdt * np.sqrt(len(subspace)/self.dim))
        #                 subproto._dict[explain_label[i]] = subproto_dict[explain_label[i]][subspace_key]
        #                 subproto.assign(explain_data[i][subspace], explain_label[i], i - self.train_size)
        #                 subproto_dict[explain_label[i]][subspace_key] = subproto._dict[explain_label[i]]
        #
        #             else:
        #                 # 存在标签，但是不存在对应的特征集合：创建特征集合的列表
        #                 subproto_dict[explain_label[i]][subspace_key] = []
        #                 subproto = Incremental_Prototype(self.ipdt * np.sqrt(len(subspace)/self.dim))
        #                 subproto.assign(explain_data[i][subspace], explain_label[i], i - self.train_size)
        #                 subproto_dict[explain_label[i]][subspace_key] = subproto._dict[explain_label[i]]
        #
        #         else:
        #             # 从头开始创建
        #             subproto_dict[explain_label[i]] = {}
        #             subproto = Incremental_Prototype(self.ipdt * np.sqrt(len(subspace)/self.dim))
        #             subproto.assign(explain_data[i][subspace], explain_label[i], i - self.train_size)
        #             subproto_dict[explain_label[i]][subspace_key] = subproto._dict[explain_label[i]]
        #
        #     explanation_lst.append(explanation)
        #
        #     print(i, ToExplain_tag, SD_diff(explanation, nbr_num=2), explanation)
        #     # for label_key in subproto_dict:
        #     #     print(label_key, [num_to_subspace(num) for num in subproto_dict[label_key]])
        #
        # for label_key in subproto_dict:
        #     for num in subproto_dict[label_key]:
        #         proto_lst = subproto_dict[label_key][num]
        #         print('label_key=', label_key, 'fea_subspace=', num_to_subspace(num), 'size=',[proto.size for proto in proto_lst])
