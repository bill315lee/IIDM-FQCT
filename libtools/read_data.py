import pandas as pd
import numpy as np
# from libtools.label_process import text_label_to_numeric_fixdata
from collections import Counter

import os


def protocol_process(protocol):
    # 0 TCP->0, udp->2

    if protocol == 'TCP':
        return 1
    elif protocol == 'UDP':
        return 0
    else:
        print('protocol_error')
        exit()



def flow_state_process(flow_state):
    # 1 L->[1,0,0,0], F->[0,1,0,0], B->[0,0,1,0], E->[0,0,0,1]
    if flow_state == 'L':
        return [1, 0, 0, 0]
    elif flow_state == 'F':
        return [0, 1, 0, 0]
    elif flow_state == 'B':
        return [0, 0, 1, 0]
    elif flow_state == 'E':
        return [0, 0, 0, 1]
    else:
        print('flow_state_error')
        exit()

def tcp_state_process(tcp_state):
    f_b = tcp_state.split('+')
    f = f_b[0].split('|')
    b = f_b[1].split('|')
    f_new = []
    b_new = []
    if 'S' in f[0]:
        f_new.append(1)
    else:
        f_new.append(0)

    if 'F' in f[1]:
        f_new.append(1)
    else:
        f_new.append(0)

    if 'R' in f[2]:
        f_new.append(1)
    else:
        f_new.append(0)

    if 'S' in b[0]:
        b_new.append(1)
    else:
        b_new.append(0)

    if 'F' in b[1]:
        b_new.append(1)
    else:
        b_new.append(0)

    if 'R' in b[2]:
        b_new.append(1)
    else:
        b_new.append(0)

    return f_new+b_new


def data_process(data):
    flow_state = []
    tcp_state = []
    protocol = []
    for i in range(len(data)):
        protocol_new = protocol_process(data[i, 0])
        flow_state_new = flow_state_process(data[i, 1])
        tcp_state_new = tcp_state_process(data[i, 2])
        protocol.append(protocol_new)
        flow_state.append(flow_state_new)
        tcp_state.append(tcp_state_new)

        # print(data[i, 0], protocol_new)
        # print(data[i, 1], flow_state_new)
        # print(data[i, 2], tcp_state_new)

    return protocol, flow_state, tcp_state




def read_toydata(date):

    data = pd.read_csv("/home/lb/project/online_adaptive_stream/toy"+date+"_new.csv", header=None).values


    return data

def read_toynewdata(date):

    data = pd.read_csv("/home/lb/project/online_adaptive_stream/toy"+date+"_new_new.csv", header=None).values


    return data





def read_IDS_online_feature(datatype, date):

    if datatype == 'IDS2017':
        path = '/mnt/IntrusionData/'+datatype+'/'+datatype+'_online_feature/'+date+'/flows_on_'+date+'.gz'
        data_all = pd.read_pickle(path).values[:, 1:]


    elif datatype == 'IDS2012':
        path = '/mnt/IntrusionData/' + datatype + '/' + datatype + '_online_feature/' + date + '/flows_on_' + date + '.gz'
        data_all = pd.read_pickle(path).values[:, 1:]
    else:
        print('datatype error')
        exit()



    four_tag = data_all[:, :4]
    time_stamp = data_all[:,5].astype(str)
    data = np.hstack((data_all[:,4].reshape(-1,1), data_all[:,7:]))

    protocol_new, tcp_state_new, flow_state_new = data_process(data)

    data_end = np.hstack((np.array(protocol_new).reshape(-1,1), np.array(flow_state_new), np.array(tcp_state_new), data[:,3:]))

    data_end = np.array(data_end, dtype=float)



    return data_end, four_tag, time_stamp

def read_IDS_fix_feature(datatype, date):

    path = '/mnt/IntrusionData/' + datatype + '/' + datatype + '_fix/' + date + '.pcap_labeling.csv'
    df = pd.read_csv(path)
    data_all = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].values[:, 1:]

    four_tag = data_all[:, :4]
    time_stamp = data_all[:, 5].astype(str)

    data = np.hstack((data_all[:, 4].reshape(-1, 1), data_all[:, 6:-1]))
    data = np.array(data,dtype=float)
    label = data_all[:, -1]
    print(Counter(label))
    if datatype == 'IDS2017':
        label = text_label_to_numeric_fixdata(label, date)
    elif datatype == 'IDS2012':
        pass
    else:
        exit()

    print(Counter(label))
    return data,label,four_tag, time_stamp


def NADC_label_process(labels):
    tag = 0
    label_tag_dict = {}
    for i in range(len(labels)):
        if labels[i] not in label_tag_dict:
            label_tag_dict[labels[i]] = tag
            tag += 1

    print(label_tag_dict)

    for i in range(len(labels)):
        labels[i] = label_tag_dict[labels[i]]

    return labels

def read_NADC(datatype, date):

    path = os.path.join('/sda', 'bill', datatype)

    df = pd.read_csv(path + '/' + date+'.csv')

    data_all = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].values

    time_stamp = data_all[:, 0].astype(str)
    data_all = data_all[:, 1:]
    four_tag = data_all[:, :4]
    data_all = data_all[:, 4:]

    data = np.hstack((data_all[:, :4], data_all[:, 5:-1])).astype(float)



    label = NADC_label_process(data_all[:, -1])
    print(Counter(label))


    np.savez(path + '/' + date + '_labelprocess', data=data, label=label, four_tag=four_tag, time_stamp=time_stamp)

    return data, label, four_tag, time_stamp


if __name__ == '__main__':
    read_NADC("NADC", "1216")



