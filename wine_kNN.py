#!/usr/bin/env python
import operator
from sklearn import cross_validation
from titanic import answer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import scale


def get_accuracies(X: np.array, Y: np.array):
    for i in range(1, 51):
        clf = KNeighborsClassifier(i)
        kf = cross_validation.KFold(len(data), 5, shuffle=True, random_state=42)
        scores = cross_validation.cross_val_score(clf, X, Y, cv=kf)
        yield scores.mean()


if __name__ == '__main__':
    data = np.genfromtxt('wine.data', delimiter=',')
    classes = data[:, 0]
    features = data[:, 1:]

    accuracies = [x for x in get_accuracies(features, classes)]
    n_neighbors, accuracy = max(enumerate(accuracies), key=operator.itemgetter(1))
    answer(n_neighbors+1, 'wine_kNN_1.txt')
    answer(accuracy, 'wine_kNN_2.txt')

    accuracies = [x for x in get_accuracies(scale(features), classes)]
    n_neighbors, accuracy = max(enumerate(accuracies), key=operator.itemgetter(1))
    answer(n_neighbors+1, 'wine_kNN_3.txt')
    answer(accuracy, 'wine_kNN_4.txt')
