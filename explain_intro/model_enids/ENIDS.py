
import numpy as np
from tqdm import tqdm

from explain_intro.model_enids.opt_explain import OPT_Explain
from base_active_classifier.Incremental_Prototypes_Weights import Incremental_Prototype_Weights
from utils import euclidian_dist, str_to_bool
import torch



class ENIDS:

    def __init__(self, args, config, dim):
        self.datatype = args.datatype
        self.train_date = args.train_date
        self.device = args.device
        self.dim = dim
        self.exp_method = args.exp_method

        self.epochs = config.epochs
        self.lr = config.lr
        self.alpha = config.alpha
        self.eps = args.eps
        self.t1 = config.t1
        self.t2 = config.t2
        self.a_ipdt = config.a_ipdt

        self.attacks_proto = Incremental_Prototype_Weights(self.a_ipdt)
        self.store_w = str_to_bool(config.store_w)
        self.batch_size = config.batch_size

        self.attack_upper_num = args.attack_upper_num





    def explain(self, explain_data, explain_label, explain_idx, proto_path):

        loss_guided_proto_idx = []
        explanation_lst = []
        explanation_idx_in_attacks = []
        epoch_lst = []
        opt_exp = OPT_Explain(epochs=self.epochs,
                                  lr=self.lr,
                                  alpha=self.alpha,
                                  eps=self.eps,
                                  t1=self.t1,
                                  t2=self.t2,
                                  dim=self.dim,
                                  device=self.device,

                                  )

        if self.store_w:
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
                    proto_name = f'batchidx_{batch_idx}'
                    normal_protos = np.load(proto_path + '/' + proto_name + '.npy', allow_pickle=True)
                    proto_array = np.array([proto for proto in normal_protos])
                    center_array = np.array([proto.center for proto in normal_protos])
                    r_array = np.array([proto.r for proto in normal_protos])

                    last_attack_batch_idx = batch_idx

                dist = euclidian_dist(explain_instance, center_array)
                cover_idx = np.where(dist <= r_array + self.eps)[0]

                if len(cover_idx) != 0:
                    continue
                else:
                    explanation_idx_in_attacks.append(i)
                    if explain_instance_label in self.attacks_proto._dict:
                        protos = self.attacks_proto._dict[explain_instance_label]
                        attack_centers = np.array([proto.center for proto in protos])
                        attack_proto_array = np.array([proto for proto in protos])
                        attack_rs = np.array([proto.r for proto in protos])
                        dist = euclidian_dist(explain_instance, attack_centers)
                        cover_idx = np.where(dist <= attack_rs + self.eps)[0]

                        if len(cover_idx) == 0:
                            w_fea_init = torch.rand(len(explain_instance)).type(torch.float).to(self.device)
                            w_proto_init = torch.rand(size=(len(center_array), len(center_array))).type(torch.float).to(
                                self.device)
                        else:
                            proto_choose = attack_proto_array[cover_idx[np.argmin(dist[cover_idx])]]
                            w_proto_init_store = proto_choose.proto_weights

                            w_fea_init = proto_choose.fea_weights
                            w_fea_init = torch.from_numpy(w_fea_init).type(torch.float).to(self.device)

                            if len(w_proto_init_store) != len(center_array):
                                w_proto_init = np.zeros((len(center_array), len(center_array)))
                                w_proto_init[:len(w_proto_init_store), :len(w_proto_init_store)] = w_proto_init_store
                            else:
                                w_proto_init = w_proto_init_store
                            w_proto_init = torch.from_numpy(w_proto_init).type(torch.float).to(self.device)

                            # print('begin init!!!!!!!!!')

                    else:
                        w_fea_init = torch.rand(len(explain_instance)).type(torch.float).to(self.device)
                        w_proto_init = torch.rand(size=(len(center_array), len(center_array))).type(torch.float).to(
                            self.device)

                    explanation, w_fea, w_proto, epoch, instance_dist = opt_exp.opt_exp(explain_instance, explain_instance_label, proto_array, w_fea_init, w_proto_init)

                    effect_idx = []
                    for j in range(len(instance_dist)):
                        if 0 in instance_dist[j]:
                            effect_idx.append(j)
                    if len(effect_idx) == 0:
                        effect_idx = [np.argmin(np.min(instance_dist, axis=1))]
                    effect_idx_array = np.array(effect_idx)

                    explanation_lst.append(explanation)
                    epoch_lst.append(epoch)
                    loss_guided_proto_idx.append(effect_idx_array)

                    self.attacks_proto.assign(explain_instance, explain_instance_label, 0, w_fea, w_proto)
                    # print('attack_proto_num',
                    #       [(label_key, len(self.attacks_proto._dict[label_key])) for label_key in self.attacks_proto._dict])

                    if len(explanation_idx_in_attacks) == self.attack_upper_num:
                        break

            attack_proto_lst = [len(self.attacks_proto._dict[label_key]) for label_key in self.attacks_proto._dict]


        else:

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
                    last_attack_batch_idx = batch_idx

                proto_array = np.array([proto for proto in normal_protos])
                center_array = np.array([proto.center for proto in normal_protos])
                r_array = np.array([proto.r for proto in normal_protos])

                dist = euclidian_dist(explain_instance, center_array)
                cover_idx = np.where(dist <= r_array + self.eps)[0]

                if len(cover_idx) != 0:
                    continue
                else:
                    explanation_idx_in_attacks.append(i)
                    w_fea_init = torch.rand(len(explain_instance)).type(torch.float).to(self.device)
                    w_proto_init = torch.rand(size=(len(center_array), len(center_array))).type(torch.float).to(
                        self.device)

                    explanation, w_fea, w_proto, epoch, instance_dist = opt_exp.opt_exp(explain_instance, explain_instance_label,
                                                                         proto_array, w_fea_init, w_proto_init)
                    effect_idx = []
                    for j in range(len(instance_dist)):
                        if 0 in instance_dist[j]:
                            effect_idx.append(j)
                    if len(effect_idx) == 0:
                        effect_idx = [np.argmin(np.min(instance_dist, axis=1))]
                    effect_idx_array = np.array(effect_idx)

                    explanation_lst.append(explanation)
                    epoch_lst.append(epoch)
                    loss_guided_proto_idx.append(effect_idx_array)

                    if len(explanation_idx_in_attacks) == self.attack_upper_num:
                        break


            attack_proto_lst = []


        return explanation_lst, np.mean(epoch_lst), explanation_idx_in_attacks, np.array(loss_guided_proto_idx), attack_proto_lst








