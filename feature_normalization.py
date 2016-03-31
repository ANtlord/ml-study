#!/usr/bin/env python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np

from titanic import answer

if __name__ == '__main__':
    train_data = np.genfromtxt('perceptron-train.csv', delimiter=',')
    test_data = np.genfromtxt('perceptron-test.csv', delimiter=',')

    X_train_data = features = train_data[:, 1:]
    Y_train_data = train_data[:, 0]
    X_test_data = features = test_data[:, 1:]
    Y_test_data = test_data[:, 0]

    scaler = StandardScaler()
    clf = Perceptron(random_state=241)

    clf.fit(X_train_data, Y_train_data)
    scores = clf.score(X_test_data, Y_test_data)
    print(scores.mean())

    X_train_data_scaled = scaler.fit_transform(X_train_data)
    X_test_data_scaled = scaler.transform(X_test_data)

    clf.fit(X_train_data_scaled, Y_train_data)
    scaled_scores = clf.score(X_test_data_scaled, Y_test_data)
    print(scores.mean(), scaled_scores.mean())
    answer(scaled_scores.mean() - scores.mean(), 'feature_normalization.txt')
