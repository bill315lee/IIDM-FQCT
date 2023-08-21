from base_active_classifier.Incremental_Prototypes import Incremental_Prototype
import argparse
import numpy as np
from tqdm import tqdm
from utils import read_train_test_data_stream
import os
from collections import Counter
from enids_config_all import *

def get_Enids_config(datatype, train_date):

    if datatype == 'NADC':
        config = NADC_Config(datatype, train_date)
    elif datatype == 'IDS2017':
        config = IDS_Config(datatype, train_date)
    elif datatype == 'creditcard':
        config = creditcard_Config(datatype, train_date)
    else:
        raise NotImplementedError
    return config


def main(args):


    print(args.datatype,'!!!!!!')
    data_use, label_use, data_mu, data_std, attack_dict = read_train_test_data_stream(args)

    if args.datatype in ['NADC', 'IDS2017']:
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{args.ipdt}_stream_{args.stream_size}_batch_{args.batch_size}_size_t{args.size_t}'
    else:
        proto_path = f'./protos_stream/{args.datatype}_ipdt_{args.ipdt}_stream_{args.stream_size}_batch_{args.batch_size}_size_t{args.size_t}'

    if not os.path.exists(proto_path):
        os.mkdir(proto_path)

    print('data_use=', data_use.shape)
    print(Counter(label_use))

    protos = Incremental_Prototype(args.ipdt)


    batch_idx = 0
    for i in range(0, len(data_use), args.batch_size):

        if i + args.batch_size > len(data_use):
            X_batch = data_use[i:len(data_use)]
            y_batch = label_use[i:len(data_use)]
        else:
            X_batch = data_use[i:i + args.batch_size]
            y_batch = label_use[i:i + args.batch_size]

        print('batch_idx=', batch_idx)
        print('X_batch=', X_batch.shape)
        print('y_batch=', Counter(y_batch))

        # 每个batch 只取前一半中的正常数据
        proto_name = f'batchidx_{batch_idx}'
        # norm_num = 0
        if args.datatype == 'IDS2017':
            for j in tqdm(range(len(X_batch))):
                if y_batch[j] == 0:
                    protos.assign(X_batch[j], y_batch[j], j + batch_idx * args.batch_size)
        elif args.datatype == 'NADC':
            idx = np.where(y_batch == 0)[0]
            # idx_random = np.random.choice(idx, 10000, replace=False)
            idx_random = idx[:10000]
            for j in tqdm(idx_random):
                if y_batch[j] == 0:
                    protos.assign(X_batch[j], y_batch[j], j + batch_idx * args.batch_size)
        else:
            for j in tqdm(range(len(X_batch))):
                if y_batch[j] == 0:
                    protos.assign(X_batch[j], y_batch[j], j + batch_idx * args.batch_size)

        # latest_idx = len(X_batch) - 1 + batch_idx * args.batch_size
        before_len = len(protos._dict[0])
        proto_lst = []
        for p in protos._dict[0]:
            if p.size >= args.size_t:
                proto_lst.append(p)

        protos._dict[0] = proto_lst

        # proto_lst_new = []
        # decay_idx = []
        # for idx, p in enumerate(protos._dict[0]):
        #     if (latest_idx - p.recent_idx) > 5 * args.batch_size:
        #         decay_idx.append(idx)
        #     else:
        #         proto_lst_new.append(p)
        # protos._dict[0] = proto_lst_new

        # print(f'Before Merge={before_len}, After Delete={len(proto_lst)}, After Decay={len(proto_lst_new)}')
        print(f'Before Merge={before_len}, After Delete={len(proto_lst)}')

        np.save(proto_path + '/' + proto_name, protos._dict[0])
        batch_idx += 1




if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--datatype', type=str, help="Dataset type.", default='NYTimes')
    arg_parser.add_argument('--train_date', type=str, help="train data date", default='1216')
    arg_parser.add_argument('--ipdt', type=float, help="threshold of proto", default=5)
    arg_parser.add_argument('--size_t', type=int, help="threshold of size", default=5)

    arg_parser.add_argument('--eps', type=float, help="error rate", default=1e-6)

    # toy


    # NADC
    # stream_size 2300000 if NADC1203;
    # stream_size 2100000 if NADC1210;
    # stream_size 1800000 if NADC1216;
    # batch_size 50000

    # IDS2017
    # stream_size 460000 if IDS2017_Tuesday;
    # stream_size 660000 if IDS2017_Wednesday;
    # stream_size 500000 if IDS2017_Thursday;
    # stream_size 680000 if IDS2017_Friday;
    # batch_size 20000
    arg_parser.add_argument('--stream_size', type=int, help='', default=1800000)
    arg_parser.add_argument('--batch_size', type=int, help="", default=10000)

    parsed_args = arg_parser.parse_args()



    main(parsed_args)




