from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score,confusion_matrix

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, precision_score, f1_score,recall_score

import numpy as np

def evaluate_acc_f1(truth, preds, average='macro'):
    acc = accuracy_score(truth, preds)
    precision = precision_score(y_true=truth, y_pred=preds, average=average)
    recall = recall_score(y_true=truth, y_pred=preds, average=average)
    f1 = f1_score(y_true=truth, y_pred=preds, average=average)
    cm = confusion_matrix(y_true=truth, y_pred=preds, labels=sorted(list(np.unique(truth))))




    return acc, precision, recall, f1, cm



def roc_auc(label, socre):
    return roc_auc_score(label, socre)

def pr(label, score):
    ap = average_precision_score(label, score)
    return ap

def ptopk(label, score):

    y_pred = get_label_n(label, score)

    return precision_score(label, y_pred)


def get_label_n(y, y_pred, n=None):
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = np.percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def best_f1(label, score):
    thre = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    f1_lst = []
    for j in range(len(thre)):
        percentile_value = np.percentile(score, thre[j])
        y_predict = []
        for i in range(len(score)):
            if score[i] > percentile_value:
                y_predict.append(1)
            else:
                y_predict.append(0)
        f1_lst.append(f1_score(label, y_predict, average="macro"))
    return np.max(f1_lst)