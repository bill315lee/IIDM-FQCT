import os
from explain_stream.model_deepaid.Model_DeepAID import Model_DeepAID
from explain_stream.model_deepaid.Deepaid_rule import Deepaid_rule_generation
from explain_stream.model_aton.Model_Aton import Model_Aton
from explain_stream.model_shap.Model_Shap import Model_Shap
from explain_stream.model_acekl.Model_acekl import Model_acekl
from explain_stream.model_enids.ENIDS import ENIDS
from explain_stream.model_enids_feature_uniform.ENIDS_fu import ENIDS_fu
from explain_stream.model_enids.Enids_rule import Enids_rule_generation
from explain_stream.model_explainer.Explainer import Explainer
from explain_stream.model_skoperules.Model_SkopeRules import Skoprules

from libtools.save_results_all import save_feature_results_enids, save_feature_results_other, save_rule_results
import numpy as np
import time
from joblib import Parallel, delayed
from utils import read_train_test_data_stream, get_feature_names, \
    rule_process
from evaluation.feature_rule_evaluate_all import stream_rule_evaluate, stream_feature_evaluate_ids
import torch

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
from compara_config import get_comparative_parser




def get_proto_param(datatype, train_date):
    proto_size_t = 10
    if datatype == 'NADC':
        ipdt = 0.5
        batch_size = 50000
        if train_date == '1203':
            stream_size = 750000
        elif train_date == '1210':
            stream_size = 1150000
        elif train_date == '1216':
            stream_size = 1800000
        else:
            raise NotImplementedError

    elif datatype == 'IDS2017':
        ipdt = 1.0
        batch_size = 20000
        if train_date == 'Tuesday':
            stream_size = 460000
        elif train_date == 'Wednesday':
            stream_size = 100000
        elif train_date == 'Thursday':
            stream_size = 500000
        elif train_date == 'Friday':
            stream_size = 240000
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return stream_size, batch_size, ipdt, proto_size_t

class Config:

    def __init__(self, args):
        self.a_ipdt = args.a_ipdt
        self.epochs = args.epochs
        self.lr = args.lr
        self.alpha = args.alpha
        self.t1 = args.t1
        self.t2 = args.t2
        self.store_w = args.store_w

        self.loss_guided = args.loss_guided
        self.keypoints = args.keypoints
        self.percentile = args.percentile
        self.tolerance = args.tolerance



        stream_size, batch_size, ipdt, proto_size_t = get_proto_param(args.datatype, args.train_date)

        self.stream_size = stream_size
        self.batch_size = batch_size
        self.ipdt = ipdt
        self.proto_size_t = proto_size_t



def main(args):

    print("This is", args.exp_method, args.explain_flag, args.datatype, args.train_date, "!!!!!")

    config = Config(args)

    print('Enids Param=')
    print(
        f'a_ipdt={config.a_ipdt}, '
        f'epochs={config.epochs}, '
        f'lr={config.lr}, '
        f'alpha={config.alpha}, '
        f't1={config.t1}, '
        f't2={config.t2}, '
        f'store_w={config.store_w}, '
        f'feat={args.feat}, '
        f'loss_guided={config.loss_guided}, '
        f'keypoints={config.keypoints}, '
        f'percentile={config.percentile}, '
        f'tolerance={config.tolerance}'
    )

    # 1.Load Data and Label
    feature_names = get_feature_names(args.datatype)
    data_use, label_use, data_mu, data_std, attack_dict = read_train_test_data_stream(args, config)

    dim = data_use.shape[1]

    feature_num_name = f'Enids_{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}' \
                       f'_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
    feature_num_path = os.path.join('.', 'feature_num')

    proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{config.stream_size}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    attacks_idx = np.where(label_use != 0)[0]
    attacks_data = data_use[attacks_idx]
    attacks_label = label_use[attacks_idx]

    if args.exp_method == 'Enids':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

        rule_path = os.path.join('.', 'feature_weight_stream', 'Enids_r')
        # rule_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_rule_num'
        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_time'

    elif args.exp_method == 'Enids_fu':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

    elif args.exp_method == 'Deepaid':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

        rule_path = os.path.join('.', 'feature_weight_stream', 'Deepaid_r')
        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_lossguide_{config.loss_guided}_kpoints_{config.keypoints}_percentile_{config.percentile}_toler_{config.tolerance}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_time'

    elif args.exp_method == 'Aton':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

    elif args.exp_method == 'Shap':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)
    elif args.exp_method == 'ACE_KL':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

    elif args.exp_method == 'Lime':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)

    elif args.exp_method == 'Explainer':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', 'Enids')

        rule_path = os.path.join('.', 'feature_weight_stream', 'Explainer')

        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_time'

    elif args.exp_method == 'Skrules':
        feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream', 'Enids')

        rule_path = os.path.join('.', 'feature_weight_stream', 'Skrules')

        rule_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_f'
        rule_inequality_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_ineq'
        rule_threshold_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_thresh'
        rule_time_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_time'

    # elif args.exp_method == 'Anchor':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
    #                           f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', 'Enids')
    #     rule_path = os.path.join('.', 'feature_weight_stream', 'Explainer')
    #     rule_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_rule_num'
    else:
        raise NotImplementedError



    # ------------------------features--------------------------------
    if args.explain_flag == 'feature':
        if not os.path.exists(feature_weight_path):
            os.makedirs(feature_weight_path)
        # 2. Explain attacks
        if not os.path.exists(feature_weight_path + '/' + feature_weight_name + '.npy'):
            time1 = time.time()
            if args.exp_method == 'Enids':
                enids = ENIDS(args, config, dim)
                f_imp_lst, epoch_mean, explained_idx_in_attacks, loss_guided_protos, attack_proto_num = enids.explain(
                    attacks_data, attacks_label, attacks_idx, proto_path)

                np.save(feature_weight_path + '/' + feature_weight_name + '_lgp.npy', loss_guided_protos)
                np.save(feature_weight_path + '/' + feature_weight_name + '_epoch.npy', epoch_mean)
                np.save(feature_weight_path + '/' + feature_weight_name + '_aproto_num.npy', attack_proto_num)

                # fea_num_dict = {}
                # feat = args.feat
                # imp_fea_num_lst = []
                # for i in range(len(f_imp_lst)):
                #     feature_importance = f_imp_lst[i]
                #     imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                #     if len(imp_idx) == 0:
                #         imp_idx = np.array([np.argmax(feature_importance)])
                #     imp_fea_num_lst.append(len(imp_idx))
                # fea_num_dict[feat] = imp_fea_num_lst
                # np.save(feature_num_path + '/' + feature_num_name, fea_num_dict)

            elif args.exp_method == 'Enids_fu':
                enids_fu = ENIDS_fu(args, config, dim)
                f_imp_lst, epoch_mean, explained_idx_in_attacks, loss_guided_protos, attack_proto_num = enids_fu.explain(
                    attacks_data, attacks_label, attacks_idx, proto_path)

                np.save(feature_weight_path + '/' + feature_weight_name + '_lgp.npy', loss_guided_protos)
                np.save(feature_weight_path + '/' + feature_weight_name + '_epoch.npy', epoch_mean)
                np.save(feature_weight_path + '/' + feature_weight_name + '_aproto_num.npy', attack_proto_num)

            elif args.exp_method == 'Deepaid':
                model_deepaid = Model_DeepAID(args, config, dim)
                f_imp_lst, explained_idx_in_attacks = model_deepaid.fit(attacks_data, attacks_label, attacks_idx,
                                                                        proto_path, feature_names)

            elif args.exp_method == 'Aton':
                model_aton = Model_Aton(args, config)
                f_imp_lst, explained_idx_in_attacks = model_aton.fit(args, attacks_data, attacks_label, attacks_idx,
                                                                     proto_path)

            elif args.exp_method == 'Shap':
                model_shap = Model_Shap(args, config)
                f_imp_lst, explained_idx_in_attacks = model_shap.fit(attacks_data, attacks_label, attacks_idx,
                                                                     proto_path)

            elif args.exp_method == 'ACE_KL':
                model_acekl = Model_acekl(args, config)
                f_imp_lst, explained_idx_in_attacks = model_acekl.fit(attacks_data, attacks_label, attacks_idx,
                                                                     proto_path)

            else:
                raise NotImplementedError
            time2 = time.time()
            print('Explained_num=', len(f_imp_lst))
            if len(f_imp_lst) == 0:
                print('wrong:', 'f_imp_lst=', len(f_imp_lst))
                exit()

            time_explain = round(time2-time1, 3)
            np.save(feature_weight_path + '/' + feature_weight_name + '.npy', np.array(f_imp_lst))
            np.save(feature_weight_path + '/' + feature_weight_name + '_idx.npy', np.array(explained_idx_in_attacks))
            np.save(feature_weight_path + '/' + feature_weight_name + '_time.npy', time_explain)
            print('time_explain=', time_explain)
            exit()
        else:
            f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
            explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')

            if args.exp_method in ['Enids', 'Enids_fu'] :
                epoch_mean = np.load(feature_weight_path + '/' + feature_weight_name + '_epoch.npy')
                attack_proto_num = np.load(feature_weight_path + '/' + feature_weight_name + '_aproto_num.npy')

            time_explain = np.load(feature_weight_path + '/' + feature_weight_name + '_time.npy')

            if args.exp_method == 'Enids':
                fea_num_dict = {}
                feat = args.feat
                imp_fea_num_lst = []
                for i in range(len(f_imp_array)):
                    feature_importance = f_imp_array[i]
                    imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                    if len(imp_idx) == 0:
                        imp_idx = np.array([np.argmax(feature_importance)])
                    imp_fea_num_lst.append(len(imp_idx))
                fea_num_dict[feat] = imp_fea_num_lst
            else:
                enids_feature_path = os.path.join('.', 'feature_weight_stream', 'Enids')
                enids_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
                enids_feature_array = np.load(enids_feature_path + '/' + enids_feature_name + '.npy',
                                              allow_pickle=True)
                fea_num_dict = {}
                feat = args.feat
                imp_fea_num_lst = []
                for i in range(len(enids_feature_array)):
                    feature_importance = enids_feature_array[i]
                    imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                    if len(imp_idx) == 0:
                        imp_idx = np.array([np.argmax(feature_importance)])
                    imp_fea_num_lst.append(len(imp_idx))
                fea_num_dict[feat] = imp_fea_num_lst



        # # 3.存储特征个数
        # if args.exp_method == 'Enids':
        #     fea_num_dict = np.load(feature_num_path + '/' + feature_num_name + '.npy', allow_pickle=True).item()
        #
        # else:
        #     fea_num_dict = np.load(feature_num_path + '/' + feature_num_name + '.npy', allow_pickle=True).item()

        in_circle_feat_lst, imp_fea_num_feat_lst = stream_feature_evaluate_ids(args, config, f_imp_array,
                                                                               explained_idx_in_attacks_array,
                                                                               args.feat, attacks_data,
                                                                               attacks_label, attacks_idx, proto_path,
                                                                               data_use, label_use)

        print(f'exp_method={args.exp_method}, time=', "%.2f" % time_explain)
        print('in_circle_num=', [round(x, 3) for x in in_circle_feat_lst])
        print('avg_imp_fea_num=', [round(x, 3) for x in imp_fea_num_feat_lst])
        cross_boundary_ratio_lst = [round(x, 3) for x in in_circle_feat_lst]
        feature_num_lst = [round(x, 3) for x in imp_fea_num_feat_lst]

        if args.exp_method == 'Enids':
            mean_epochs = np.round(epoch_mean, 3)
            print('Mean_Epochs_Num=', mean_epochs)
            save_feature_results_enids(args, config, cross_boundary_ratio_lst, feature_num_lst, time_explain, mean_epochs,
                                       len(f_imp_array), attack_proto_num)
        else:
            save_feature_results_other(args, config, cross_boundary_ratio_lst, feature_num_lst, time_explain)


    # ----------------rule------------------------
    elif args.explain_flag == 'rule':

        rule_feature_save_file = os.path.join(rule_path, rule_feature_name)
        rule_ineq_save_file = os.path.join(rule_path, rule_inequality_name)
        rule_threshold_save_file = os.path.join(rule_path, rule_threshold_name)
        rule_time_save_file = os.path.join(rule_path, rule_time_name)

        explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
        explained_data = attacks_data[explained_idx_in_attacks_array]
        explained_label = attacks_label[explained_idx_in_attacks_array]
        explained_idx = attacks_idx[explained_idx_in_attacks_array]

        if not os.path.exists(rule_feature_save_file+'.npy'):

            if not os.path.exists(rule_path):
                os.makedirs(rule_path)

            if args.exp_method == 'Enids':
                f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
                LGP_idx = np.load(feature_weight_path + '/' + feature_weight_name + '_lgp.npy', allow_pickle=True)

                if args.exp_method == 'Enids':
                    fea_num_dict = {}
                    feat = args.feat
                    imp_fea_num_lst = []
                    for i in range(len(f_imp_array)):
                        feature_importance = f_imp_array[i]
                        imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                        if len(imp_idx) == 0:
                            imp_idx = np.array([np.argmax(feature_importance)])
                        imp_fea_num_lst.append(len(imp_idx))
                    fea_num_dict[feat] = imp_fea_num_lst

                else:
                    enids_feature_path = os.path.join('.', 'feature_weight_stream', 'Enids')
                    enids_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
                    enids_feature_array = np.load(enids_feature_path + '/' + enids_feature_name + '.npy',
                                                  allow_pickle=True)
                    fea_num_dict = {}
                    feat = args.feat
                    imp_fea_num_lst = []
                    for i in range(len(enids_feature_array)):
                        feature_importance = enids_feature_array[i]
                        imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                        if len(imp_idx) == 0:
                            imp_idx = np.array([np.argmax(feature_importance)])
                        imp_fea_num_lst.append(len(imp_idx))
                    fea_num_dict[feat] = imp_fea_num_lst

                    # for i in range(len(enids_feature_array)):
                    #     imp_idx = np.arange(dim)
                    #     imp_fea_num_lst.append(imp_idx)
                    # fea_num_dict[feat] = imp_fea_num_lst

                feature_lst_explain = [np.argsort(-f_imp_array[i])[:fea_num_dict[args.feat][i]] for i in range(len(fea_num_dict[args.feat]))]

                time1 = time.time()
                feature_lst_explain, inequality_lst_explain, threshold_lst_explain = Enids_rule_generation(feature_lst_explain,
                                                                                                           LGP_idx,
                                                                                                           explained_data,
                                                                                                           explained_label,
                                                                                                           explained_idx,
                                                                                                           proto_path,
                                                                                                           config.batch_size,
                                                                                                           config.loss_guided,
                                                                                                           config.keypoints,
                                                                                                           config.percentile,
                                                                                                           config.tolerance,
                                                                                                           args.eps)


                time2 = time.time()
            elif args.exp_method == 'Deepaid':

                f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
                enids_rule_path = os.path.join('.', 'feature_weight_stream', 'Enids_r')
                enids_feature_name = f'{args.datatype}{args.train_date}_ipdt_{config.ipdt}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}'
                enids_rule_feature_array = np.load(enids_rule_path + '/' + enids_feature_name + '.npy',
                                                   allow_pickle=True)
                rule_num = [len(feature_lst) for feature_lst in enids_rule_feature_array]
                time1 = time.time()
                feature_lst_explain, inequality_lst_explain, threshold_lst_explain = Deepaid_rule_generation(
                    f_imp_array, rule_num, explained_data)
                time2 = time.time()


            elif args.exp_method == 'Explainer':

                time1 = time.time()
                explainer = Explainer(args, config)
                feature_lst_explain, inequality_lst_explain, threshold_lst_explain, rule_num_lst = \
                    explainer.fit(explained_data, explained_label, explained_idx, proto_path)

                time2 = time.time()

            elif args.exp_method == 'Skrules':
                time1 = time.time()
                clf = Skoprules(feature_names, config.batch_size)
                feature_lst_explain, inequality_lst_explain, threshold_lst_explain = clf.explain(explained_data,
                                                                                                 explained_label,
                                                                                                 explained_idx,
                                                                                                 proto_path)
                time2 = time.time()


            else:
                raise NotImplementedError

            feature_lst_explain, inequality_lst_explain, threshold_lst_explain = \
                rule_process(feature_lst_explain, inequality_lst_explain, threshold_lst_explain)

            time_explain = time2 - time1



            np.save(rule_feature_save_file, np.array(feature_lst_explain))
            np.save(rule_ineq_save_file, np.array(inequality_lst_explain))
            np.save(rule_threshold_save_file, np.array(threshold_lst_explain))
            np.save(rule_time_save_file, round(time_explain, 3))
            exit()
        else:

            feature_lst_explain = np.load(rule_feature_save_file+'.npy', allow_pickle=True)
            inequality_lst_explain = np.load(rule_ineq_save_file+'.npy', allow_pickle=True)
            threshold_lst_explain = np.load(rule_threshold_save_file+'.npy', allow_pickle=True)
            time_explain = np.load(rule_time_save_file+'.npy')


        print('Begin Rule Evaluation')
        results = Parallel(n_jobs=-1)(
            delayed(stream_rule_evaluate)(args,
                                          config,
                                          explained_data[i],
                                          explained_label[i],
                                          explained_idx[i],
                                          explained_data,
                                          explained_label,
                                          feature_lst_explain[i],
                                          inequality_lst_explain[i],
                                          threshold_lst_explain[i],
                                          proto_path,
                                          data_use,
                                          label_use) for i in range(len(explained_data)))

        roc_lst = []
        pr_lst = []
        f1_lst = []
        attack_precision_lst = []
        attack_recall_lst = []
        for i in range(len(results)):
            roc_lst.append(results[i][0])
            pr_lst.append(results[i][1])
            f1_lst.append(results[i][2])
            attack_precision_lst.append(results[i][3])
            attack_recall_lst.append(results[i][4])

        roc_mean = np.mean(roc_lst)
        pr_mean = np.mean(pr_lst)
        f1_mean = np.mean(f1_lst)
        attack_precision_mean = np.mean(attack_precision_lst)
        attack_recall_mean = np.mean(attack_recall_lst)

        fea_len = 0
        for i in range(len(feature_lst_explain)):
            fea_len += len(set(feature_lst_explain[i]))
        fea_len = fea_len / len(feature_lst_explain)
        print(f'This is {args.exp_method}')
        print('roc_auc_mean_value=', round(roc_mean, 3))
        print('pr_mean_value=', round(pr_mean, 3))
        print('f1_mean_value=', round(f1_mean, 3))
        print('attack_precsion_mean_value=', round(attack_precision_mean, 3))
        print('attack_recall_mean_value=', round(attack_recall_mean, 3))
        print('avg_fea_len=', round(fea_len, 3))
        print('time_explain=', time_explain)

        save_rule_results(args, config, round(roc_mean, 3), round(pr_mean, 3), round(f1_mean, 3),
                          round(attack_precision_mean, 3), round(attack_recall_mean, 3),
                          round(fea_len, 3), len(explained_data), time_explain)

if __name__ == '__main__':
    arg_parser = get_comparative_parser()

    arg_parser.add_argument('--datatype', type=str, help="Dataset type.", default='IDS2017')
    arg_parser.add_argument('--train_date', type=str, help="train data date", default='Tuesday')
    arg_parser.add_argument('--device', type=str, help="Torch device.", default='cuda:0',
                            choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    arg_parser.add_argument('--exp_method', type=str, help="train data date", default='Enids')
    arg_parser.add_argument('--eps', type=float, help="error rate", default=1e-6)
    arg_parser.add_argument('--explain_flag', type=str, help="", default="feature")
    arg_parser.add_argument('--feat', type=float, help="", default=0.9)



    # feature
    arg_parser.add_argument('--a_ipdt', type=float, help="", default=1.0)
    arg_parser.add_argument('--epochs', type=int, help="", default=500)
    arg_parser.add_argument('--lr', type=float, help="", default=0.1)
    arg_parser.add_argument('--alpha', type=float, help="", default=1)
    arg_parser.add_argument('--t1', type=float, help="", default=0.5)
    arg_parser.add_argument('--t2', type=float, help="", default=0.1)
    arg_parser.add_argument('--store_w', type=str, help="", default='True')


    # rule
    arg_parser.add_argument('--loss_guided', type=str, help="", default='True')
    arg_parser.add_argument('--keypoints', type=str, help="", default='False')
    arg_parser.add_argument('--percentile', type=float, help="", default=10.0)
    arg_parser.add_argument('--tolerance', type=float, help="", default=0.5)






    parsed_args = arg_parser.parse_args()


    main(parsed_args)
