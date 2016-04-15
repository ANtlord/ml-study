#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
from math import exp
from math import log
from titanic import answer
from scipy.spatial.distance import euclidean

from sklearn.metrics import roc_auc_score

THRESHOLD = 1e-5
LEARNING_RATE = 0.1  # k
REGULARIZATION_COEFF = 10


def _log_arg(y, x1, x2, w1_in, w2_in):
    return 1 + exp(-y * (w1_in * x1 + w2_in * x2))


def _algorithm(x1: float, x2: float, w1_in: float, w2_in: float):  # a(x)
    return 1 / (1 + exp(-w1_in*x1 - w2_in*x2))


def _loss_function(y, x1, x2, w1_in, w2_in):
    return log(_log_arg(y, x1, x2, w1_in, w2_in))


def _diff_base(y, x1, x2, w1_in, w2_in):
    return 1 - (1 / _log_arg(y, x1, x2, w1_in, w2_in))


def _l2_regularisation(regularization_coeff: float, w1_in: float, w2_in: float):
    """
    (1/2)*(C||w||^2)
    :param regularization_coeff: C
    :param w1_in: w1
    :param w2_in: w2
    :return:
    """
    return (regularization_coeff * (abs(norm([w1_in, w2_in])) ** 2)) / 2


def _compute_emperical_risk(objects: np.ndarray, results: np.ndarray, w1_in: float, w2_in: float,
                            regularization_coeff: float):
    objects_len = objects.shape[0]  # l
    functional_result = sum(
        [_loss_function(y, xi[0], xi[1], w1_in, w2_in) for xi, y in zip(objects, results)]
    ) / objects_len
    return functional_result + _l2_regularisation(regularization_coeff, w1_in, w2_in)


def _compute_weights(x_train_data, y_train_data, regularization_coeff):
    weight1, weight2 = 0, 0
    l = y_train_data.shape[0]
    # emperical_risk = _compute_emperical_risk(x_train_data, y_train_data, weight1, weight2,
    #                                          regularization_coeff)
    for _ in range(10000):
        w1_new = (
            weight1 +
            (LEARNING_RATE / l) * sum([y * xi[0] * _diff_base(y, xi[0], xi[1], weight1, weight2)
                                       for xi, y in zip(x_train_data, y_train_data)]) -
            LEARNING_RATE * regularization_coeff * weight1
        )
        w2_new = (
            weight2 +
            (LEARNING_RATE / l) * sum([y * xi[1] * _diff_base(y, xi[0], xi[1], weight1, weight2)
                                       for xi, y in zip(x_train_data, y_train_data)]) -
            LEARNING_RATE * regularization_coeff * weight2
        )

        if euclidean([weight1, weight2], [w1_new, w2_new]) <= THRESHOLD:
            return weight1, weight2

        weight1, weight2 = w1_new, w2_new
        # emperical_risk = _compute_emperical_risk(x_train_data, y_train_data, weight1, weight2,
        #                                          regularization_coeff)
    return weight1, weight2


if __name__ == '__main__':
    train_data = np.genfromtxt('data-logistic.csv', delimiter=',')
    X_train_data = train_data[:, 1:]
    Y_train_data = train_data[:, 0]
    w1, w2 = _compute_weights(X_train_data, Y_train_data, REGULARIZATION_COEFF)
    w1_, w2_ = _compute_weights(X_train_data, Y_train_data, 0)
    answer('%s %s' % (
        roc_auc_score(Y_train_data, [_algorithm(x[0], x[1], w1_, w2_) for x in X_train_data]),
        roc_auc_score(Y_train_data, [_algorithm(x[0], x[1], w1, w2) for x in X_train_data])
    ), 'logistic_res.txt')
