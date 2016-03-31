#!/usr/bin/env python
import operator
from titanic import answer

from sklearn import datasets
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
import numpy as np

if __name__ == '__main__':
    data = datasets.load_boston()
    X = scale(data.data)
    accuracies = {}
    for i in np.linspace(1, 10, 200):
        knr = KNeighborsRegressor(weights='distance', p=i)
        kf = cross_validation.KFold(data.data.shape[0], 5, shuffle=True, random_state=42)
        scores = cross_validation.cross_val_score(knr, X, data.target, cv=kf,
                                                  scoring='mean_squared_error')
        accuracies[i] = scores.mean()
    best_p, accuracy = max(accuracies.items(), key=operator.itemgetter(1))
    answer(best_p, 'boston_metric.txt')
