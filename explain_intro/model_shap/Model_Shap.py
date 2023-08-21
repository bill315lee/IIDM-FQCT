
import numpy as np
from tqdm import tqdm
from utils import predictor_train
from explain_intro.model_shap.SHAP import SHAP
from utils import euclidian_dist, str_to_bool
from collections import Counter

class Model_Shap:

    def __init__(self, args, config):
        self.batch_size = config.batch_size
        self.eps = args.eps
        self.attack_dict = {}
        self.train_all_data = str_to_bool(args.train_all_data)
        self.attack_upper_num = args.attack_upper_num

    def fit(self, explain_data, explain_label, explain_idx, proto_path, data_use):

        f_imp_lst = []
        last_attack_batch_idx = 0
        explanation_idx_in_attacks = []
        for i in tqdm(range(len(explain_data))):

            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]
            explain_instance_idx = explain_idx[i]
            batch_idx = int(explain_instance_idx / self.batch_size) - 1
            if batch_idx <= 0:
                continue
            if batch_idx == last_attack_batch_idx:
                pass
            else:
                proto_name = f'batchidx_{batch_idx}.npy'
                normal_protos = np.load(proto_path + '/' + proto_name, allow_pickle=True)



                proto_array = np.array([proto for proto in normal_protos])
                center_array = np.array([proto.center for proto in normal_protos])
                r_array = np.array([proto.r for proto in normal_protos])

                last_attack_batch_idx = batch_idx
            dist = euclidian_dist(explain_instance, center_array)
            cover_idx = np.where(dist <= r_array + self.eps)[0]

            if len(cover_idx) != 0:
                continue
            else:
                if self.train_all_data:
                    idx_lst = sum([proto.data_idx_lst for proto in proto_array], [])
                    idx_array = np.array(idx_lst)
                    pass_data = data_use[idx_array]


                    if explain_instance_label in self.attack_dict:
                        attacks = np.array(self.attack_dict[explain_instance_label])
                        attack_labels = np.ones(len(self.attack_dict[explain_instance_label]))

                        inputs = np.vstack((pass_data, attacks, explain_instance.reshape(1, -1)))
                        targets = np.array(list(np.zeros(len(pass_data))) + list(attack_labels) + [1], dtype=int)
                        sizes = np.array(list(np.ones(len(pass_data))) + list(attack_labels) + [1])
                    else:
                        inputs = np.vstack((pass_data, explain_instance.reshape(1, -1)))
                        targets = np.array(list(np.zeros(len(pass_data))) + [1], dtype=int)
                        sizes = np.array(list(np.ones(len(pass_data))) + [1])

                else:

                    if explain_instance_label in self.attack_dict:
                        attacks = np.array(self.attack_dict[explain_instance_label])
                        attack_labels = np.ones(len(self.attack_dict[explain_instance_label]))

                        inputs = np.vstack((center_array, attacks, explain_instance.reshape(1, -1)))
                        targets = np.array(list(np.zeros(len(center_array))) + list(attack_labels) + [1], dtype=int)
                        sizes = np.array(list(np.ones(len(center_array))) + list(attack_labels) + [1])
                    else:
                        inputs = np.vstack((center_array, explain_instance.reshape(1, -1)))
                        targets = np.array(list(np.zeros(len(center_array))) + [1], dtype=int)
                        sizes = np.array(list(np.ones(len(center_array))) + [1])

                predictor = predictor_train(inputs, targets, sizes)
                model_shap = SHAP(n_sample=100)
                f_imp_array = model_shap.fit(inputs, targets, explain_instance.reshape(1,-1), 1, predictor)

                if explain_instance_label in self.attack_dict:
                    self.attack_dict[explain_instance_label].append(explain_instance)
                else:
                    self.attack_dict[explain_instance_label] = [explain_instance]

                f_imp_lst.append(f_imp_array)
                explanation_idx_in_attacks.append(i)

                if len(explanation_idx_in_attacks) == self.attack_upper_num:
                    break

        return f_imp_lst, explanation_idx_in_attacks