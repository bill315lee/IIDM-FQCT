import pandas as pd
import numpy as np
from explain_method.model_lore.util import *
from collections import Counter


def prepare_dataset(df, args):
    # Features Categorization
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]

    class_name = 'class'
    possible_outcomes = list(df[class_name].unique())



    if args.datatype == "NADC":
        # df['app'] = df['app'].astype('object')
        # df['proto'] = df['proto'].astype('object')
        df['class'] = df['class'].astype('object')
    elif args.datatype == "IDS2017":
        exit()
    else:
        exit()

    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)

    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'


    # discrete = ['proto', 'app']
    # continuous = ["duration","out_bytes", "in_bytes",
    #              "cnt_dst", "cnt_src", "cnt_serv_src","cnt_serv_dst","cnt_dst_slow","cnt_src_slow",
    #              "cnt_serv_src_slow","cnt_serv_dst_slow","cnt_dst_conn","cnt_src_conn","cnt_serv_src_conn","cnt_serv_dst_conn"]

    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=None, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    X = df.values[:,:-1]
    y = df.values[:,-1]

    dataset = {
        'name': f'{args.datatype}_{args.train_date}',
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'X': X,
        'y': y,
    }


    return dataset