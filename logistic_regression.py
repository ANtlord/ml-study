#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
from math import exp
from math import log
from random import randint


def _diff_base(y, x1, x2, w1_in, w2_in):
    return 1 - (1 / 1 - exp(-y * (w1_in * x1 + w2_in * x2)))


def _score_base(y, x1, x2, w1_in, w2_in):
    return log(1 + exp(-y * (w1_in * x1 + w2_in * x2)))


if __name__ == '__main__':
    train_data = np.genfromtxt('data-logistic.csv', delimiter=',')
    X_train_data = train_data[:, 1:]
    Y_train_data = train_data[:, 0]

    train_data_len = len(train_data)
    w1, w2 = 0, 0
    k = 0.1
    c = 10
    l = len(train_data)
    score = sum(_score_base(x[0], x[1], x[2], w1, w2) for x in train_data) / l + \
        (c * pow(abs(np.linalg.norm([w1, w2])), 2)) / 2

    i_his = []
    for _ in range(10000):
        i = randint(0, l - 1)
        while i in i_his:
            i = randint(0, l - 1)
        lost_score = _score_base(Y_train_data[i], X_train_data[i][0], X_train_data[i][1], w1, w2)
        x = train_data[i]
        w1_new = w1 - k * x[0] * x[1] * _diff_base(x[0], x[1], x[2], w1, w2)
        w2_new = w2 - k * x[0] * x[1] * _diff_base(x[0], x[1], x[2], w1, w2)

        # w1_new = w1 + (k / l * sum(x[0] * x[1] * _diff_base(x[0], x[1], x[2], w1, w2)
        #                            for x in train_data)) - k * c * w1
        # w2_new = w2 + (k / l * sum(x[0] * x[2] * _diff_base(x[0], x[1], x[2], w1, w2)
        #                            for x in train_data)) - k * c * w2

        if abs(norm([w1_new, w2_new]) - norm([w1, w2])) <= 1e-5:
            break

        score = (1 - 1 / k) * score + lost_score / k
        print('score', score)

        w1, w2 = round(w1_new, 3), round(w2_new, 3)
        # w1, w2 = w1_new, w2_new
        print(w1, w2)
        i_his.append(i)

    print(w1, w2)
