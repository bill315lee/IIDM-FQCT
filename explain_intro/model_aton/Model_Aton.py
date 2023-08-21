import numpy as np
from tqdm import tqdm
from explain_intro.model_aton.ATON import ATON
from utils import euclidian_dist, str_to_bool


class Model_Aton:
    def __init__(self, args, config):
        self.eps = args.eps
        self.batch_size = config.batch_size
        self.train_all_data = str_to_bool(args.train_all_data)
        self.attack_upper_num = args.attack_upper_num

    def fit(self, args, explain_data, explain_label, explain_idx, proto_path, data_use):

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
            cover_idx = np.where(dist <= r_array + self.eps)[0]
            if len(cover_idx) != 0:
                continue
            else:

                model_aton = ATON(verbose=False, gpu=True,
                                  nbrs_num=args.nbrs_aton, rand_num=args.rand_aton,
                                  alpha1=args.alpha1_aton, alpha2=args.alpha2_aton,
                                  n_epoch=args.epoch_aton, batch_size=args.batchsize_aton, lr=args.lr_aton,
                                  n_linear=args.nlinear_aton, margin=args.margin_aton)

                if self.train_all_data:
                    idx_lst = sum([proto.data_idx_lst for proto in proto_array], [])
                    idx_array = np.array(idx_lst)
                    pass_data = data_use[idx_array]
                    input = np.vstack((pass_data, explain_instance.reshape(1, -1)))
                    targets = np.array(list(np.zeros(len(pass_data))) + [1], dtype=int)
                else:
                    input = np.vstack((center_array, explain_instance.reshape(1, -1)))
                    targets = np.array(list(np.zeros(len(center_array))) + [1], dtype=int)




                f_imp_array = model_aton.fit(input, targets)[0]
                f_imp_lst.append(f_imp_array)
                explanation_idx_in_attacks.append(i)

                if len(explanation_idx_in_attacks) == self.attack_upper_num:
                    break


        return f_imp_lst, explanation_idx_in_attacks