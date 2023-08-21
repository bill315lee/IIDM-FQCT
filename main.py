import os
from explain_stream.model_deepaid.Model_DeepAID import Model_DeepAID
from explain_stream.model_deepaid.Deepaid_rule import Deepaid_rule_generation
from explain_stream.model_aton.Model_Aton import Model_Aton
from explain_stream.model_shap.Model_Shap import Model_Shap
from explain_stream.model_enids.ENIDS import ENIDS
from explain_stream.model_anchor.Anchor import Anchor_rule
from explain_stream.model_explainer.Explainer import Explainer
from libtools.save_results import save_feature_results_enids, save_feature_results_other, save_rule_results
import numpy as np
import time
from joblib import Parallel, delayed
from utils import euclidian_dist, read_train_test_data_stream, predictor_train, str_to_bool, get_feature_names, \
    rule_process
from evaluation.feature_rule_evaluate_toy import stream_rule_evaluate, stream_feature_evaluate_ids, \
    stream_normal_rule_evaluate
import torch

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
from compara_config import get_comparative_parser
from enids_config import NADC_Config, IDS_Config





def get_Enids_config(datatype, train_date):

    if datatype == 'NADC':
        config = NADC_Config(datatype, train_date)
    elif datatype == 'IDS2017':
        config = IDS_Config(datatype, train_date)
    else:
        raise NotImplementedError
    return config


def main(args):

    print("This is", args.exp_method, args.explain_flag, args.datatype, args.train_date, args.attack_label, "!!!!!")

    config = get_Enids_config(args.datatype, args.train_date)

    print('Enids Param=')
    print(
        f'a_ipdt={config.a_ipdt}, '
        f'epochs={config.epochs}, '
        f'lr={config.lr}, '
        f'alpha={config.alpha}, '
        f't1={config.t1}, '
        f't2={config.t2}, '
        f'core_t={config.core_t}, '
        f'store_w={config.store_w}, '
        f'loss_guided={config.loss_guided}, '
        f'correctify={config.correctify},'
        f'reparam_f={args.reparam_f},'
        f'reparam_p={args.reparam_p}'
    )

    # 1.Load Data and Label
    feature_names = get_feature_names(args.datatype)
    data_use, label_use, data_mu, data_std, attack_dict = read_train_test_data_stream(args)
    dim = data_use.shape[1]

    feature_num_name = f'Enids_{args.datatype}{args.train_date}_aipdt{config.a_ipdt}_epochs{config.epochs}_lr{config.lr}_alpha{config.alpha}' \
                       f'_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_reparamf_{args.reparam_f}_reparamp_{args.reparam_p}'
    feature_num_path = os.path.join('.', 'feature_num_reparam')

    if args.train_date == '1203':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{750000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == '1210':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{1150000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == '1216':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{1800000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == 'Tuesday':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{460000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == 'Wednesday':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{100000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == 'Thursday':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{500000}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    elif args.train_date == 'Friday':
        proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{240000}_batch_{config.batch_size}_size_t{config.proto_size_t}'


    # proto_path = f'./protos_stream/{args.datatype}_{args.train_date}_ipdt_{config.ipdt}_stream_{config.stream_size}_batch_{config.batch_size}_size_t{config.proto_size_t}'
    attacks_idx = np.where(label_use == args.attack_label)[0]
    attacks_data = data_use[attacks_idx]
    attacks_label = label_use[attacks_idx]

    if args.exp_method == 'Enids':
        feature_weight_name = f'{args.datatype}{args.train_date}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
                              f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_reparamf_{args.reparam_f}_reparamp_{args.reparam_p}'
        feature_weight_path = os.path.join('.', 'feature_weight_stream_reparam', args.exp_method)

        rule_num_path = os.path.join('.', 'feature_weight_stream', 'Explainer')
        rule_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_reparamf_{args.reparam_f}_reparamp_{args.reparam_p}_rule_num'


    # elif args.exp_method == 'Deepaid':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)
    #     rule_num_path = os.path.join('.', 'feature_weight_stream', 'Explainer')
    #     rule_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_rule_num'
    #
    # elif args.exp_method == 'Aton':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)
    #
    # elif args.exp_method == 'Shap':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)
    # elif args.exp_method == 'Lime':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', args.exp_method)
    #
    # elif args.exp_method == 'Explainer':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
    #                           f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', 'Enids')
    #
    #     rule_num_path = os.path.join('.', 'feature_weight_stream', 'Explainer')
    #     rule_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_rule_num'
    #
    # elif args.exp_method == 'Anchor':
    #     feature_weight_name = f'{args.datatype}{args.train_date}_aipdt{config.a_ipdt}_epochs{config.epochs}' \
    #                           f'_lr{config.lr}_alpha{config.alpha}_t1_{config.t1}_t2_{config.t2}_storew_{config.store_w}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}'
    #     feature_weight_path = os.path.join('.', 'feature_weight_stream', 'Enids')
    #     rule_num_path = os.path.join('.', 'feature_weight_stream', 'Explainer')
    #     rule_name = f'{args.datatype}{args.train_date}_a_label_{args.attack_label}_a_uppernum_{args.attack_upper_num}_rule_num'
    # else:
    #     raise NotImplementedError

    if not os.path.exists(feature_weight_path):
        os.makedirs(feature_weight_path)

    # ------------------------features--------------------------------
    if args.explain_flag == 'feature':
        # 2. Explain attacks
        time1 = time.time()
        if not os.path.exists(feature_weight_path + '/' + feature_weight_name + '.npy'):
            if args.exp_method == 'Enids':
                enids = ENIDS(args, config, dim)
                f_imp_lst, epoch_mean, explained_idx_in_attacks, loss_guided_protos, attack_proto_num = enids.explain(
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
            else:
                raise NotImplementedError

            print('Explained_num=', len(f_imp_lst))
            if len(f_imp_lst) == 0:
                exit()
            f_imp_array = np.array(f_imp_lst)
            explained_idx_in_attacks_array = np.array(explained_idx_in_attacks)
            np.save(feature_weight_path + '/' + feature_weight_name + '.npy', f_imp_array)
            np.save(feature_weight_path + '/' + feature_weight_name + '_idx.npy', explained_idx_in_attacks_array)

        else:
            f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
            explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
            if args.exp_method == 'Enids':
                epoch_mean = np.load(feature_weight_path + '/' + feature_weight_name + '_epoch.npy')
                attack_proto_num = np.load(feature_weight_path + '/' + feature_weight_name + '_aproto_num.npy')
        time2 = time.time()

        # 3.存储特征个数
        if args.exp_method == 'Enids':
            if os.path.exists(feature_num_path + '/' + feature_num_name):
                fea_num_dict = np.load(feature_num_path + '/' + feature_num_name + '.npy', allow_pickle=True).item()
            else:
                fea_num_dict = {}
                feat_lst = args.fea_lst
                for feat in feat_lst:
                    imp_fea_num_lst = []
                    for i in range(len(f_imp_array)):
                        feature_importance = f_imp_array[i]
                        imp_idx = np.where(feature_importance >= feat - args.eps)[0]
                        if len(imp_idx) == 0:
                            imp_idx = np.array([np.argmax(feature_importance)])
                        imp_fea_num_lst.append(len(imp_idx))
                    fea_num_dict[feat] = imp_fea_num_lst
                np.save(feature_num_path + '/' + feature_num_name, fea_num_dict)
        else:
            fea_num_dict = np.load(feature_num_path + '/' + feature_num_name + '.npy', allow_pickle=True).item()
        in_circle_feat_lst, imp_fea_num_feat_lst = stream_feature_evaluate_ids(args, config, f_imp_array,
                                                                               explained_idx_in_attacks_array,
                                                                               fea_num_dict, attacks_data,
                                                                               attacks_label, attacks_idx, proto_path,
                                                                               data_use, label_use)

        time_explain = round((time2 - time1), 3)

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

    elif args.explain_flag == 'rule':

        if not os.path.exists(rule_num_path):
            os.makedirs(rule_num_path)

        fea_num_dict = np.load(feature_num_path + '/' + feature_num_name + '.npy', allow_pickle=True).item()

        if args.exp_method == 'Enids':
            f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
            explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')

            rule_num = np.load(rule_num_path + '/' + rule_name + '.npy')
            LGP_idx = np.load(feature_weight_path + '/' + feature_weight_name + '_lgp.npy', allow_pickle=True)
            feature_lst_explain = [np.argsort(-f_imp_array[i])[:rule_num[i]] for i in range(len(rule_num))]
            explained_data = attacks_data[explained_idx_in_attacks_array]
            explained_idx = attacks_idx[explained_idx_in_attacks_array]

            results = Parallel(n_jobs=20)(
                delayed(stream_normal_rule_evaluate)(args, config,
                                                     explained_data[i],
                                                     explained_data,
                                                     LGP_idx[i],
                                                     feature_lst_explain[i],
                                                     explained_idx[i],
                                                     proto_path,
                                                     data_use,
                                                     label_use) for i in range(len(explained_data)))
            roc_lst = []
            pr_lst = []
            f1_lst = []
            attack_recall_lst = []
            for i in range(len(results)):
                roc_lst.append(results[i][0])
                pr_lst.append(results[i][1])
                f1_lst.append(results[i][2])
                attack_recall_lst.append(results[i][3])

            roc_mean = np.mean(roc_lst)
            pr_mean = np.mean(pr_lst)
            f1_mean = np.mean(f1_lst)
            attack_recall_mean = np.mean(attack_recall_lst)


        # elif args.exp_method == 'Deepaid':
        #     f_imp_array = np.load(feature_weight_path + '/' + feature_weight_name + '.npy')
        #     explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
        #
        #     rule_num = np.load(rule_num_path + '/' + rule_name + '.npy')
        #
        #     explained_data = attacks_data[explained_idx_in_attacks_array]
        #     feature_lst_explain, inequality_lst_explain, threshold_lst_explain = \
        #         Deepaid_rule_generation(f_imp_array, rule_num, explained_data)
        #
        #     print('Begin Rule Evaluation')
        #     roc_mean, pr_mean, f1_mean, attack_recall_mean = stream_rule_evaluate(args,config,
        #                                                                           explained_idx_in_attacks_array,
        #                                                                           attacks_data, attacks_label,
        #                                                                           attacks_idx, proto_path,
        #                                                                           data_use, label_use,
        #                                                                           feature_lst_explain,
        #                                                                           inequality_lst_explain,
        #                                                                           threshold_lst_explain)
        #
        # elif args.exp_method == 'Explainer':
        #
        #     explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
        #     explained_data = attacks_data[explained_idx_in_attacks_array]
        #     explained_label = attacks_label[explained_idx_in_attacks_array]
        #     explained_idx = attacks_idx[explained_idx_in_attacks_array]
        #
        #     explainer = Explainer(args, config)
        #     feature_lst_explain, inequality_lst_explain, threshold_lst_explain, rule_num_lst = \
        #         explainer.fit(explained_data, explained_label, explained_idx, proto_path, fea_num_dict[args.fea_lst[0]])
        #
        #     feature_lst_explain, inequality_lst_explain, threshold_lst_explain = \
        #         rule_process(feature_lst_explain, inequality_lst_explain, threshold_lst_explain)
        #
        #     np.save(os.path.join(rule_num_path, rule_name), np.array(rule_num_lst))
        #
        #     print('Begin Rule Evaluation')
        #     roc_mean, pr_mean, f1_mean, attack_recall_mean = stream_rule_evaluate(args,config,
        #                                                                           explained_idx_in_attacks_array,
        #                                                                           attacks_data, attacks_label,
        #                                                                           attacks_idx, proto_path,
        #                                                                           data_use, label_use,
        #                                                                           feature_lst_explain,
        #                                                                           inequality_lst_explain,
        #                                                                           threshold_lst_explain)
        #
        #
        # elif args.exp_method == 'Anchor':
        #
        #     explained_idx_in_attacks_array = np.load(feature_weight_path + '/' + feature_weight_name + '_idx.npy')
        #     explained_data = attacks_data[explained_idx_in_attacks_array]
        #     explained_label = attacks_label[explained_idx_in_attacks_array]
        #     explained_idx = attacks_idx[explained_idx_in_attacks_array]
        #
        #     rule_num = np.load(rule_num_path + '/' + rule_name + '.npy')
        #     model = Anchor_rule(args, config)
        #     feature_lst_explain, inequality_lst_explain, threshold_lst_explain \
        #         = model.fit(explained_data, explained_label, explained_idx, proto_path, rule_num, feature_names)
        #
        #     print('Begin Rule Evaluation')
        #     roc_mean, pr_mean, f1_mean, attack_recall_mean = stream_rule_evaluate(args,config,
        #                                                                           explained_idx_in_attacks_array,
        #                                                                           attacks_data, attacks_label,
        #                                                                           attacks_idx, proto_path,
        #                                                                           data_use, label_use,
        #                                                                           feature_lst_explain,
        #                                                                           inequality_lst_explain,
        #                                                                           threshold_lst_explain)
        # else:
        #     raise NotImplementedError

        fea_len = 0
        for i in range(len(feature_lst_explain)):
            fea_len += len(set(feature_lst_explain[i]))
        fea_len = fea_len / len(feature_lst_explain)

        print(f'This is {args.exp_method}')
        print('roc_auc_mean_value=', round(roc_mean, 3))
        print('pr_mean_value=', round(pr_mean, 3))
        print('f1_mean_value=', round(f1_mean, 3))
        print('attack_recall_mean_value=', round(attack_recall_mean, 3))
        print('avg_fea_len=', round(fea_len, 3))


        save_rule_results(args, config, round(roc_mean, 3), round(pr_mean, 3), round(f1_mean, 3), round(attack_recall_mean, 3), round(fea_len, 3), len(explained_data))


if __name__ == '__main__':
    arg_parser = get_comparative_parser()

    arg_parser.add_argument('--datatype', type=str, help="Dataset type.", default='toy')
    arg_parser.add_argument('--train_date', type=str, help="train data date", default='1')
    arg_parser.add_argument('--device', type=str, help="Torch device.", default='cuda:0',
                            choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    arg_parser.add_argument('--exp_method', type=str, help="train data date", default='Enids')
    arg_parser.add_argument('--eps', type=float, help="error rate", default=1e-6)
    arg_parser.add_argument('--explain_flag', type=str, help="", default="feature")
    arg_parser.add_argument('--attack_label', type=int, help="", default=1)
    arg_parser.add_argument('--attack_upper_num', type=int, help="", default=100)
    arg_parser.add_argument('--fea_lst', type=list, help="", default=[0.9])
    arg_parser.add_argument('--reparam_f', type=str, help="", default="False")
    arg_parser.add_argument('--reparam_p', type=str, help="", default="False")



    parsed_args = arg_parser.parse_args()


    main(parsed_args)
