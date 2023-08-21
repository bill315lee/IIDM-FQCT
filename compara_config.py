import argparse


def get_basic_parser():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--device', type=str, help="Torch device.", default='cuda:0',
                            choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    arg_parser.add_argument('--print_iter', type=int, default=10000)
    arg_parser.add_argument('--train_size', type=int, help="", default=100000)
    arg_parser.add_argument('--batch_size', type=int, help="", default=256)
    arg_parser.add_argument('--query_ratio', type=float, help="", default=0.01)
    arg_parser.add_argument('--normal_threshold', type=float, help="", default=0.99)
    arg_parser.add_argument('--verbose', type=str, default='True')

    return arg_parser


def get_comparative_parser():

    arg_parser = argparse.ArgumentParser()

    # anchor
    arg_parser.add_argument('--t_anchor', type=float, help="threshold of anchor", default=0.99)

    # Deepaid
    arg_parser.add_argument('--t_deepaid', type=float, help="", default=0.99)
    arg_parser.add_argument('--epoch_deepaid', type=int, help="", default=100)
    arg_parser.add_argument('--lr_deepaid', type=float, help="", default=0.1)
    arg_parser.add_argument('--batchsize_deepaid', type=int, help="", default=1024)
    arg_parser.add_argument('--steps_deepaid', type=int, help="", default=100)

    # aton
    arg_parser.add_argument('--nbrs_aton', type=int, help="", default=30)
    arg_parser.add_argument('--rand_aton', type=int, help="", default=30)
    arg_parser.add_argument('--alpha1_aton', type=float, help="", default=0.8)
    arg_parser.add_argument('--alpha2_aton', type=float, help="", default=0.2)
    arg_parser.add_argument('--epoch_aton', type=int, help="", default=50)
    arg_parser.add_argument('--batchsize_aton', type=int, help="", default=128)
    arg_parser.add_argument('--lr_aton', type=float, help="", default=0.01)
    arg_parser.add_argument('--nlinear_aton', type=int, help="", default=64)
    arg_parser.add_argument('--margin_aton', type=int, help="", default=5)

    # explainer
    arg_parser.add_argument('--treesnum_explainer', type=int, help="", default=10)
    arg_parser.add_argument('--sample_explainer', type=int, help="", default=128)

    return arg_parser