
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from explain_method.model_enids.opt_explain import OPT_Explain
from base_active_classifier.Incremental_Prototypes_Weights import Incremental_Prototype_Weights
from utils import euclidian_dist, str_to_bool
import torch




class ENIDS:

    def __init__(self, args, dim):
        self.datatype = args.datatype
        self.train_date = args.train_date
        self.device = args.device
        self.dim = dim
        self.exp_method = args.exp_method

        self.epochs = args.epochs
        self.lr = args.lr
        self.alpha = args.alpha
        self.eps = args.eps
        self.t1 = args.t1
        self.t2 = args.t2
        self.reparam_f = str_to_bool(args.reparam_f)
        self.reparam_p = str_to_bool(args.reparam_p)

    def explain(self, explain_data, explain_label, proto_dict):
        proto_array = np.array([proto for proto in proto_dict[0]])

        explanation_lst = []
        epoch_lst = []
        loss_guided_proto_idx = []
        opt_exp = OPT_Explain(epochs=self.epochs,
                                  lr=self.lr,
                                  alpha=self.alpha,
                                  eps=self.eps,
                                  t1=self.t1,
                                  t2=self.t2,
                                  dim=self.dim,
                                  device=self.device,
                              reparam_f= self.reparam_f,
                              reparam_p = self.reparam_p

                                  )

        for i in tqdm(range(len(explain_data))):
            explain_instance = explain_data[i]
            explain_instance_label = explain_label[i]
            explanation, epoch, instance_dist = opt_exp.opt_exp(explain_instance, explain_instance_label, proto_array)

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




        return explanation_lst, np.mean(epoch_lst), np.array(loss_guided_proto_idx)








