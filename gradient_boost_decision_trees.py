#!/usr/bin/env python
import numpy as np
import pandas
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from titanic import answer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from math import exp
from matplotlib.pyplot import savefig
from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import legend

from sklearn.metrics import log_loss

if __name__ == '__main__':
    data_train = pandas.read_csv('gbm-data.csv', index_col=None)  # type: DataFrame
    X_data_train = data_train.values[:, 1:]
    Y_data_train = data_train.values[:, 0]
    arrays = train_test_split(X_data_train, Y_data_train, test_size=0.8, random_state=241)
    X_data_train = arrays[0]
    X_data_test = arrays[1]
    Y_data_train = arrays[2]
    Y_data_test = arrays[3]

    answer2_argmin = None
    answer2_value = None
    for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
        clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241,
                                         learning_rate=learning_rate)
        clf.fit(X_data_train, Y_data_train)

        train_probs = clf.predict_proba(X_data_train)
        test_probs = clf.predict_proba(X_data_test)

        train_losts = []
        for pred in clf.staged_decision_function(X_data_train):
            train_losts.append(log_loss(Y_data_train, [1 / (1 + exp(-x)) for x in pred]))
        train_losts = np.array(train_losts)

        test_losts = []
        for pred in clf.staged_decision_function(X_data_test):
            test_losts.append(log_loss(Y_data_test, [1 / (1 + exp(-x)) for x in pred]))
        test_losts = np.array(test_losts)

        figure()
        plot(test_losts, 'g', linewidth=2)
        plot(train_losts, 'r', linewidth=2)
        legend(['test', 'train'])
        savefig('image-%s.png' % learning_rate)

        if learning_rate == 0.2:
            answer2_argmin = np.argmin(test_losts)
            answer2_value = test_losts.min()

    f_clf = RandomForestClassifier(random_state=241, n_estimators=answer2_argmin)
    f_clf.fit(X_data_train, Y_data_train)
    rf_min_loss = log_loss(Y_data_test, f_clf.predict_proba(X_data_test))

    answer('overfitting', 'gradient_boost_decision_trees-1.txt')
    answer('%s %s' % (answer2_value, answer2_argmin), 'gradient_boost_decision_trees-2.txt')
    answer(rf_min_loss, 'gradient_boost_decision_trees-3.txt')
