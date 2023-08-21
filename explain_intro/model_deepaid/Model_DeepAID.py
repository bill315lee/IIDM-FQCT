from explain_intro.model_deepaid.deepaid import DeepAID
from explain_intro.model_deepaid.autoencoder import train
from tqdm import tqdm
import numpy as np
from utils import euclidian_dist, str_to_bool
from collections import Counter

class Model_DeepAID:

    def __init__(self, args, config, dim):
        self.args = args
        self.dim = dim
        self.batch_size = config.batch_size
        self.steps_deepaid = args.steps_deepaid
        self.train_all_data = str_to_bool(args.train_all_data)
        self.attack_upper_num = args.attack_upper_num

    def fit(self, explain_data, explain_label, explain_idx, proto_path, feature_names, data_use):
        explanation_idx_in_attacks = []
        f_imp_lst = []
        last_attack_batch_idx = 0
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
            cover_idx = np.where(dist <= r_array + self.args.eps)[0]
            if len(cover_idx) != 0:
                continue
            else:

                if self.train_all_data:
                    idx_lst = sum([proto.data_idx_lst for proto in proto_array], [])
                    idx_array = np.array(idx_lst)
                    pass_data = data_use[idx_array]
                    ae_model, thres = train(pass_data, self.dim, self.args)
                else:
                    ae_model, thres = train(center_array, self.dim, self.args)
                model = DeepAID(ae_model, thres, self.dim, feature_desc=feature_names, steps=self.steps_deepaid)
                fea_imp = model(explain_instance)
                f_imp_lst.append(fea_imp)
                explanation_idx_in_attacks.append(i)

                if len(explanation_idx_in_attacks) == self.attack_upper_num:
                    break


        return f_imp_lst, explanation_idx_in_attacks

