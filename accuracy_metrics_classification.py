#!/usr/bin/env python
import numpy as np
import pandas
from titanic import answer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

if __name__ == '__main__':
    data = pandas.read_csv('classification.csv')
    true_positive = data[data['true'] == 1][data['pred'] == 1]
    false_positive = data[data['true'] == 0][data['pred'] == 1]
    false_negative = data[data['true'] == 1][data['pred'] == 0]
    true_negative = data[data['true'] == 0][data['pred'] == 0]

    tp_count = true_positive.shape[0]
    fp_count = false_positive.shape[0]
    fn_count = false_negative.shape[0]
    tn_count = true_negative.shape[0]
    answer('%s %s %s %s' % (tp_count, fp_count, fn_count, tn_count),
           'accuracy_metrics_classification_1.txt')

    accuracy = (tp_count + tn_count) / sum([tp_count, fp_count, fn_count, tn_count])
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f_score = f1_score(data['true'], data['pred'])
    answer('%s %s %s %s' % (accuracy, precision, recall, f_score),
           'accuracy_metrics_classification_2.txt')

    scores = pandas.read_csv('scores.csv')
    roc_auc_scores = dict(score_logreg=roc_auc_score(scores['true'], scores['score_logreg']),
                          score_svm=roc_auc_score(scores['true'], scores['score_svm']),
                          score_knn=roc_auc_score(scores['true'], scores['score_knn']),
                          score_tree=roc_auc_score(scores['true'], scores['score_tree']))

    answer('%s' % (max(roc_auc_scores.items(), key=lambda x: x[1])[0]),
           'accuracy_metrics_classification_3.txt')

    max_precisions = {}
    for key, value in roc_auc_scores.items():
        precision, recall, thresholds = precision_recall_curve(scores['true'], scores[key])
        prt_array = np.vstack((precision, recall))
        prt_array = prt_array.transpose()
        prt_array = prt_array[prt_array[:, 1] >= 0.7]
        max_precisions[key] = prt_array.max(axis=0)[0]

    answer('%s' % (max(max_precisions.items(), key=lambda x: x[1])[0]),
           'accuracy_metrics_classification_4.txt')
