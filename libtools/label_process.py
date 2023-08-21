import time
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime
from datetime import timedelta

# timediff = 39600
timediff = timedelta(hours=3)
DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def IDS2012_labels(train_date, four_tag_lst, time_stamp):
    attack_file_path = '/mnt/IntrusionData/IDS2012/IDS2012_pcap/five_key_timestamp/'
    if train_date == '0612':
        attack_file_name = 'SatJun12Flows.csv'
        label = get_0612_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)

    elif train_date == '0613':
        attack_file_name = 'SunJun13Flows.csv'
        label = get_0613_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)
    elif train_date == '0614':
        attack_file_name = 'MonJun14Flows.csv'
        label = get_0614_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)
    elif train_date == '0615':
        attack_file_name = 'TueJun15Flows.csv'
        label = get_0615_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)
    elif train_date == '0616':
        attack_file_name = 'WedJun16Flows.csv'
        label = get_0616_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)
    elif train_date == '0617':
        attack_file_name = 'ThuJun17Flows.csv'
        label = get_0617_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name)

    else:
        print('Date Error')
        exit()


    return np.array(label, dtype=int)


def time_process(str):


    time_lst = str.split(':')
    a = [int(item) for item in time_lst]

    hour = a[0]
    min = a[1]
    sec = a[2]



    if min == 0:
        tt_min = 3600*(hour - 1) + 60*59+ sec
    else:
        tt_min = 3600 * hour + 60 * (min-1) + sec

    if min == 59:
        tt_max = 3600 * (hour + 1) + 60 * 0 + sec
    else:
        tt_max = 3600 * hour + 60*(min+1) + sec

    return tt_min, tt_max

def get_0612_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):

    attack_data = pd.read_csv(attack_file_path+attack_file_name, header=None).values
    attack_key = attack_data[:,2:]
    label = []
    for i in range(len(time_stamp)):

        attack_flag = False
        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]
        srcPort = four_tag_lst[i][1]
        destPort = four_tag_lst[i][3]

        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))
        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute

        if hour == 11:
            if minutes >= 33 and minutes <= 35:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j,0] and srcPort == attack_key[j,1] and destIP == attack_key[j,2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break
            if minutes >= 40 and minutes <= 42:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[j, 2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break

        if hour==16:
            if minutes >= 22:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j,0] and srcPort == attack_key[j,1] and destIP == attack_key[j,2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break

        if hour>= 17 and hour<= 19:
            for j in range(len(attack_key)):
                if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                    j, 2] and destPort == attack_key[j, 3]:
                    attack_flag = True
                    break


        if hour == 20:
            if (minutes <=6) or (minutes >= 45 and minutes <= 52):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[j, 2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break



        if hour == 22:
            if minutes >= 39  and minutes <= 41:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[j, 2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break
        if hour == 23:
            if minutes >= 16  and minutes <= 18:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[j, 2] and destPort == attack_key[j,3]:
                        attack_flag = True
                        break

        if attack_flag == False:
            label.append(0)
        else:
            label.append(1)



    return label


def get_0613_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):

    attack_data = pd.read_csv(attack_file_path + attack_file_name, header=None).values
    attack_start_time = attack_data[:, 0]
    attack_end_time = attack_data[:, 1]
    attack_four_tag = attack_data[:, 2:]

    # 构建hash表
    hash_dict = {}

    for i in range(len(attack_data)):
        hash_value = hash(str(attack_four_tag[i]))
        tt_min = time_process(attack_start_time[i])[0]
        tt_max = time_process(attack_end_time[i])[1]
        if hash_value in hash_dict:
            hash_dict[hash_value].append([tt_min, tt_max])
        else:
            hash_dict[hash_value] = [[tt_min, tt_max]]


    label = np.zeros(len(time_stamp))

    for i in range(len(time_stamp)):
        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))


        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute
        second = start_time.second

        flow_time = 3600*hour+60*minutes+second

        hash_value = hash(str(np.array(four_tag_lst[i])))
        if hash_value in hash_dict:
            time_lst = hash_dict[hash_value]
            if four_tag_lst[i][0] == '192.168.1.105' and int(four_tag_lst[i][1])==34431:
                print(four_tag_lst[i])
                print(start_time, flow_time, time_lst)

            for t_min_max in time_lst:
                if (flow_time>=t_min_max[0]) and (flow_time<=t_min_max[1]):
                    label[i] = 1
                    break


    return label


def get_0614_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):
    attack_data = pd.read_csv(attack_file_path + attack_file_name, header=None).values
    attack_start_time = attack_data[:, 0]
    attack_end_time = attack_data[:, 1]
    attack_four_tag = attack_data[:, 2:]

    # 构建hash表
    hash_dict = {}

    for i in range(len(attack_data)):
        hash_value = hash(str(attack_four_tag[i]))
        tt_min = time_process(attack_start_time[i])[0]
        tt_max = time_process(attack_end_time[i])[1]
        if hash_value in hash_dict:
            hash_dict[hash_value].append([tt_min, tt_max])
        else:
            hash_dict[hash_value] = [[tt_min, tt_max]]

    label = np.zeros(len(time_stamp))

    for i in range(len(time_stamp)):
        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))

        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute
        second = start_time.second

        flow_time = 3600 * hour + 60 * minutes + second

        hash_value = hash(str(np.array(four_tag_lst[i])))
        if hash_value in hash_dict:
            time_lst = hash_dict[hash_value]
            for t_min_max in time_lst:
                if (flow_time >= t_min_max[0]) and (flow_time <= t_min_max[1]):
                    label[i] = 1
                    break

    return label


def get_0615_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):
    attack_data = pd.read_csv(attack_file_path + attack_file_name, header=None).values
    attack_start_time = attack_data[:, 0]
    attack_end_time = attack_data[:, 1]
    attack_four_tag = attack_data[:, 2:]

    # 构建hash表
    hash_dict = {}

    for i in range(len(attack_data)):
        hash_value = hash(str(attack_four_tag[i]))
        tt_min = time_process(attack_start_time[i])[0]
        tt_max = time_process(attack_end_time[i])[1]
        if hash_value in hash_dict:
            hash_dict[hash_value].append([tt_min, tt_max])
        else:
            hash_dict[hash_value] = [[tt_min, tt_max]]

    label = np.zeros(len(time_stamp))

    for i in range(len(time_stamp)):
        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))

        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute
        second = start_time.second

        flow_time = 3600 * hour + 60 * minutes + second

        hash_value = hash(str(np.array(four_tag_lst[i])))
        if hash_value in hash_dict:
            time_lst = hash_dict[hash_value]
            for t_min_max in time_lst:
                if (flow_time >= t_min_max[0]) and (flow_time <= t_min_max[1]):
                    label[i] = 1
                    break

    return label

def get_0616_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):
    attack_data = pd.read_csv(attack_file_path + attack_file_name, header=None).values
    attack_key = attack_data[:, 2:]
    label = []
    for i in range(len(time_stamp)):

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]
        srcPort = four_tag_lst[i][1]
        destPort = four_tag_lst[i][3]

        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))
        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute


        attack_flag = False

        if hour == 10:
            if minutes <= 46 and minutes >= 44:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if hour == 14:
            if minutes <= 37 and minutes >= 35:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break


        if hour == 15:
            if (minutes <= 17 and minutes >= 15) or (minutes <= 42 and minutes >= 39):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if hour == 16:
            if (minutes <= 4 and minutes >= 2) or (minutes <= 16 and minutes >= 14) or (minutes <= 23 and minutes >= 21):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if hour == 17:
            if (minutes <= 5 and minutes >= 3):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break
        if hour == 18:
            if (minutes <= 14 and minutes >= 12) or (minutes <= 55 and minutes >= 53):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if hour == 19:
            if (minutes <= 16 and minutes >= 14):
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if attack_flag == False:
            label.append(0)

        else:
            label.append(1)


    return label


def get_0617_label(four_tag_lst, time_stamp, attack_file_path, attack_file_name):
    attack_data = pd.read_csv(attack_file_path + attack_file_name, header=None).values
    attack_key = attack_data[:, 2:]
    label = []
    for i in range(len(time_stamp)):

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]
        srcPort = four_tag_lst[i][1]
        destPort = four_tag_lst[i][3]

        t_split = time_stamp[i].rsplit(':', 1)
        t = t_split[0] + ':' + str('%f' % float(t_split[1]))
        start_time = datetime.strptime(t, DATE_FORMAT) - timediff
        hour = start_time.hour
        minutes = start_time.minute

        attack_flag = False

        if (hour == 14 and minutes >= 25) or (hour == 15) or (hour == 16 and minutes <= 47):
            for j in range(len(attack_key)):
                if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                    j, 2] and destPort == attack_key[j, 3]:
                    attack_flag = True
                    break

        if hour == 17:
            if minutes <= 14 and minutes >= 11:
                for j in range(len(attack_key)):
                    if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                        j, 2] and destPort == attack_key[j, 3]:
                        attack_flag = True
                        break

        if (hour == 21 and minutes >=36) and (hour==22 and minutes <=1):
            for j in range(len(attack_key)):
                if srcIP == attack_key[j, 0] and srcPort == attack_key[j, 1] and destIP == attack_key[
                    j, 2] and destPort == attack_key[j, 3]:
                    attack_flag = True
                    break


        if attack_flag == False:
            label.append(0)

        else:
            label.append(1)

    return label



def IDS2017_labels(train_date, four_tag_lst, time_stamp):

    if train_date == 'Tuesday':
        label = get_Tuesday_label(four_tag_lst, time_stamp)
    elif train_date == 'Wednesday':
        label = get_Wednesday_label(four_tag_lst, time_stamp)
    elif train_date == 'Thursday':
        label = get_Thursday_label(four_tag_lst, time_stamp)
    elif train_date == 'Friday':
        label = get_Friday_label(four_tag_lst, time_stamp)
    else:
        print('Date Error')
        exit()


    label_new = text_label_to_numeric(np.array(label), train_date)

    return label_new




def text_label_to_numeric_fixdata(label, date):

    if date == 'Monday':
        label[np.where(label == 'BENIGN')[0]] = 0

    elif date == 'Tuesday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'FTP-Patator')[0]] = 1
        label[np.where(label == 'SSH-Patator')[0]] = 2

    elif date == 'Wednesday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'DoS slowloris')[0]] = 1
        label[np.where(label == 'DoS Slowhttptest')[0]] = 2
        label[np.where(label == 'DoS Hulk')[0]] = 3
        label[np.where(label == 'DoS GoldenEye')[0]] = 4
        label[np.where(label == 'Heartbleed')[0]] = 5

    # elif date == 'Thursday_1':
    #     label[np.where(label == 'BENIGN')[0]] = 0
    #     label[np.where(label == 'Web Attack - Brute Force')[0]] = 1
    #     label[np.where(label == 'Web Attack - XSS')[0]] = 2
    #     label[np.where(label == 'Web Attack - Sql Injection')[0]] = 3

    elif date == 'Thursday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'Web Attack - Brute Force')[0]] = 1
        label[np.where(label == 'Web Attack - XSS')[0]] = 1
        label[np.where(label == 'Web Attack - Sql Injection')[0]] = 1
        label[np.where(label == 'Infiltration')[0]] = 2

    elif date == 'Friday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'Bot')[0]] = 1
        label[np.where(label == 'PortScan')[0]] = 2
        label[np.where(label == 'DDoS')[0]] = 3

    else:
        print('label error')
        exit()

    return np.array(label, dtype=int)


def text_label_to_numeric(label, date):

    if date == 'Monday':
        label[np.where(label == 'BENIGN')[0]] = 0

    elif date == 'Tuesday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'FTP-Patator')[0]] = 1
        label[np.where(label == 'SSH-Patator')[0]] = 2

    elif date == 'Wednesday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'DoS Slowloris')[0]] = 1
        label[np.where(label == 'DoS Slowhttp')[0]] = 2
        label[np.where(label == 'DoS Hulk')[0]] = 3
        label[np.where(label == 'DoS GoldenEye')[0]] = 4
        label[np.where(label == 'HeartBleed')[0]] = 5

    # elif date == 'Thursday_1':
    #     label[np.where(label == 'BENIGN')[0]] = 0
    #     label[np.where(label == 'Web Attack - Brute Force')[0]] = 1
    #     label[np.where(label == 'Web Attack - XSS')[0]] = 2
    #     label[np.where(label == 'Web Attack - Sql Injection')[0]] = 3

    elif date == 'Thursday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'Web Attack-Brute Force')[0]] = 1
        label[np.where(label == 'Web Attack-XSS')[0]] = 1
        label[np.where(label == 'Web Attack-Sql Injection')[0]] = 1
        label[np.where(label == 'Infiltration')[0]] = 2

    elif date == 'Friday':
        label[np.where(label == 'BENIGN')[0]] = 0
        label[np.where(label == 'Bot')[0]] = 1
        label[np.where(label == 'PortScan')[0]] = 2
        label[np.where(label == 'DDoS')[0]] = 3

    else:
        print('label error')
        exit()


    return np.array(label, dtype=int)


def get_Tuesday_label(four_tag_lst, time_stamp):

    label = []
    for i in range(len(time_stamp)):
        hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
        minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]

        if hour == 9:
            if minutes >= 18 and minutes < 60:
                if srcIP == '172.16.0.1' and destIP == '192.168.10.50':
                    label.append("FTP-Patator")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 10:
            if minutes >= 0 and minutes <= 21:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("FTP-Patator")

                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")

        elif hour == 14:
            if minutes >= 8 and minutes < 60:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("SSH-Patator")

                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 15:
            if minutes >= 0 and minutes <= 12:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("SSH-Patator")

                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        else:
            label.append("BENIGN")

    return label

def get_Wednesday_label(four_tag_lst, time_stamp):
    label = []

    for i in range(len(time_stamp)):
        hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
        minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]

        if hour == 9:
            if minutes >= 47 and minutes < 60:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Slowloris")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")

        elif hour == 10:
            if minutes >= 0 and minutes <= 12:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Slowloris")
                else:
                    label.append("BENIGN")

            elif minutes >= 14 and minutes <= 38:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Slowhttp")
                else:
                    label.append("BENIGN")
            elif minutes >= 42 and minutes < 60:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Hulk")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")

        elif hour == 11:
            if minutes >= 0 and minutes <= 8:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Hulk")
                else:
                    label.append("BENIGN")

            elif minutes >= 9 and minutes <= 20:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS GoldenEye")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")

        elif hour == 14:
            if minutes >= 23 and minutes <= 26:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DoS Slowloris")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")


        elif hour == 15:
            if minutes >= 11 and minutes < 34:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.51":
                    label.append("HeartBleed")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        else:
            label.append("BENIGN")


    return label

def get_Thursday_label(four_tag_lst, time_stamp):
    label = []
    for i in range(len(time_stamp)):
        hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
        minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]

        if hour == 9:
            if minutes >= 14 and minutes < 60:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("Web Attack-Brute Force")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 10:
            if minutes <= 1:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("Web Attack-Brute Force")
                else:
                    label.append("BENIGN")
            elif minutes >= 14 and minutes <= 36:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("Web Attack-XSS")
                else:
                    label.append("BENIGN")
            elif minutes >= 39 and minutes <= 43:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("Web Attack-Sql Injection")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 14:
            if minutes == 19 or (minutes >= 27 and minutes < 60):
                if srcIP == "192.168.10.8" and destIP == "205.174.165.73":
                    label.append("Infiltration")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 15:
            if minutes >= 0 and minutes <= 46:
                if srcIP == "192.168.10.8" and destIP == "205.174.165.73":
                    label.append("Infiltration")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        else:
            label.append("BENIGN")

    return label

def get_Friday_label(four_tag_lst, time_stamp):
    label = []

    for i in range(len(time_stamp)):
        hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
        minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]

        srcIP = four_tag_lst[i][0]
        destIP = four_tag_lst[i][2]

        if hour == 10:
            if minutes >= 3 and minutes < 60:
                if srcIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14', '192.168.10.5',
                             '192.168.10.8'] \
                        and destIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14',
                                       '192.168.10.5', '192.168.10.8']:
                    label.append("Bot")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 9:
            if minutes >= 33 and minutes <= 36:
                if srcIP == "192.168.10.12" and destIP == "52.6.13.28":
                    label.append("Bot")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 11 or hour == 12:
            if srcIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14', '192.168.10.5',
                         '192.168.10.8'] \
                    and destIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14',
                                   '192.168.10.5',
                                   '192.168.10.8']:
                label.append("Bot")
            else:
                label.append("BENIGN")

        elif hour == 13:
            if (minutes >= 4 and minutes <= 7) or (minutes >= 51 and minutes < 60):
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("PortScan")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")

        elif hour == 14:
            if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                label.append("PortScan")
            else:
                label.append("BENIGN")
        elif hour == 15:
            if (minutes >= 0 and minutes <= 24):
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("PortScan")
                else:
                    label.append("BENIGN")
            elif minutes >= 55 and minutes < 60:
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DDoS")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        elif hour == 16:
            if (minutes >= 0 and minutes <= 17):
                if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
                    label.append("DDoS")
                else:
                    label.append("BENIGN")
            else:
                label.append("BENIGN")
        else:
            label.append("BENIGN")

    return label



# def get_Tuesday_label(four_tag_lst, time_stamp):
#
#     label = []
#     for i in range(len(time_stamp)):
#         hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
#         minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]
#
#         srcIP = four_tag_lst[i][0]
#         destIP = four_tag_lst[i][2]
#
#         if hour == 9:
#             if minutes >= 18 and minutes < 60:
#                 if srcIP == '172.16.0.1' and destIP == '192.168.10.50':
#                     label.append("FTP-Patator")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 10:
#             if minutes >= 0 and minutes <= 21:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("FTP-Patator")
#
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 14:
#             if minutes >= 8 and minutes < 60:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("SSH-Patator")
#
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 15:
#             if minutes >= 0 and minutes <= 12:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("SSH-Patator")
#
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         else:
#             label.append("BENIGN")
#
#     return label
#
# def get_Wednesday_label(four_tag_lst, time_stamp):
#     label = []
#
#     for i in range(len(time_stamp)):
#         hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
#         minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]
#
#         srcIP = four_tag_lst[i][0]
#         destIP = four_tag_lst[i][2]
#
#         if hour == 9:
#             if minutes >= 47 and minutes < 60:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Slowloris")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 10:
#             if minutes >= 0 and minutes <= 12:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Slowloris")
#                 else:
#                     label.append("BENIGN")
#
#             elif minutes >= 14 and minutes <= 38:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Slowhttp")
#                 else:
#                     label.append("BENIGN")
#             elif minutes >= 42 and minutes < 60:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Hulk")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 11:
#             if minutes >= 0 and minutes <= 8:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Hulk")
#                 else:
#                     label.append("BENIGN")
#
#             elif minutes >= 9 and minutes <= 20:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS GoldenEye")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 14:
#             if minutes >= 23 and minutes <= 26:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DoS Slowloris")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#
#         elif hour == 15:
#             if minutes >= 11 and minutes < 34:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.51":
#                     label.append("HeartBleed")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         else:
#             label.append("BENIGN")
#
#
#     return label
#
# def get_Thursday_label(four_tag_lst, time_stamp):
#     label = []
#     for i in range(len(time_stamp)):
#         hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
#         minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]
#
#         srcIP = four_tag_lst[i][0]
#         destIP = four_tag_lst[i][2]
#
#         if hour == 9:
#             if minutes >= 14 and minutes < 60:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("Web Attack-Brute Force")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 10:
#             if minutes <= 1:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("Web Attack-Brute Force")
#                 else:
#                     label.append("BENIGN")
#             elif minutes >= 14 and minutes <= 36:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("Web Attack-XSS")
#                 else:
#                     label.append("BENIGN")
#             elif minutes >= 39 and minutes <= 43:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("Web Attack-Sql Injection")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 14:
#             if minutes == 19 or (minutes >= 27 and minutes < 60):
#                 if srcIP == "192.168.10.8" and destIP == "205.174.165.73":
#                     label.append("Infiltration")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 15:
#             if minutes >= 0 and minutes <= 46:
#                 if srcIP == "192.168.10.8" and destIP == "205.174.165.73":
#                     label.append("Infiltration")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         else:
#             label.append("BENIGN")
#
#     return label
#
# def get_Friday_label(four_tag_lst, time_stamp):
#     label = []
#
#     for i in range(len(time_stamp)):
#         hour = time.localtime(np.float64(time_stamp[i][0]) - 39600)[3]
#         minutes = time.localtime(np.float64(time_stamp[i][0]) - 39600)[4]
#
#         srcIP = four_tag_lst[i][0]
#         destIP = four_tag_lst[i][2]
#
#         if hour == 10:
#             if minutes >= 3 and minutes < 60:
#                 if srcIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14', '192.168.10.5',
#                              '192.168.10.8'] \
#                         and destIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14',
#                                        '192.168.10.5', '192.168.10.8']:
#                     label.append("Bot")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 9:
#             if minutes >= 33 and minutes <= 36:
#                 if srcIP == "192.168.10.12" and destIP == "52.6.13.28":
#                     label.append("Bot")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 11 or hour == 12:
#             if srcIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14', '192.168.10.5',
#                          '192.168.10.8'] \
#                     and destIP in ['205.174.165.73', '192.168.10.15', '192.168.10.9', '192.168.10.14',
#                                    '192.168.10.5',
#                                    '192.168.10.8']:
#                 label.append("Bot")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 13:
#             if (minutes >= 4 and minutes <= 7) or (minutes >= 51 and minutes < 60):
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("PortScan")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#
#         elif hour == 14:
#             if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                 label.append("PortScan")
#             else:
#                 label.append("BENIGN")
#         elif hour == 15:
#             if (minutes >= 0 and minutes <= 24):
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("PortScan")
#                 else:
#                     label.append("BENIGN")
#             elif minutes >= 55 and minutes < 60:
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DDoS")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         elif hour == 16:
#             if (minutes >= 0 and minutes <= 17):
#                 if srcIP == "172.16.0.1" and destIP == "192.168.10.50":
#                     label.append("DDoS")
#                 else:
#                     label.append("BENIGN")
#             else:
#                 label.append("BENIGN")
#         else:
#             label.append("BENIGN")
#
#     return label

if __name__ == '__main__':
    date = "0613"
    print(date)
    data = pd.read_csv("/mnt/IntrusionData/IDS2012/IDS2012_fix/"+date+".pcap_Flow.csv").values

    four_tag_lst = data[:,1:5]
    timestamp = data[:,6]
    labels = IDS2012_labels(date, four_tag_lst, timestamp)
    print(Counter(labels))

    data_all = np.hstack((data[:,:-1], labels.reshape(-1,1)))

    np.savetxt('/mnt/IntrusionData/IDS2012/IDS2012_fix/'+date+'.pcap_labeling.csv', data_all,delimiter=',', fmt = '%s')