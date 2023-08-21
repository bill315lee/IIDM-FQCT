from tqdm import tqdm
from base_active_classifier.Incremental_Toy import Incremental_Prototype_Toy
import numpy as np

class Prototypes:
    def __init__(self, args):
        self.ipdt = args.ipdt



    def proto_fit(self, data, label):
        train_size = len(data)

        protos = Incremental_Prototype_Toy(self.ipdt)
        for i in tqdm(range(len(data))):
            protos.assign(data[i], label[i], i - train_size)

        before_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        # for label_key in set(label):
        #     protos.merge(label_key)
        after_merge_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])

        proto_lst = []
        for p in protos._dict[0]:
            if p.size > 1:
                proto_lst.append(p)
        protos._dict[0] = proto_lst


        after_delete_len = np.sum([len(protos._dict[label_key]) for label_key in protos._dict])
        print(f'Before Merge={before_len}, After Merge={after_merge_len}, After Delete={after_delete_len}')

        return protos._dict
