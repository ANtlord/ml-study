#!/usr/bin/env python
import pandas
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from titanic import answer

if __name__ == '__main__':
    data_train = pandas.read_csv('abalone.csv', index_col=None)  # type: DataFrame
    data_train['Sex'] = data_train['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    data = data_train.values[:, :-1]
    target = data_train.values[:, -1]

    i = None
    for i in range(1, 50):
        knr = RandomForestRegressor(i, random_state=1)
        kf = KFold(len(target), 5, shuffle=True, random_state=1)
        scores = cross_validation.cross_val_score(estimator=knr, X=data, y=target, scoring='r2', cv=kf)
        accuracy = scores.mean()
        if accuracy > 0.52:
            break

    answer(i, 'forest_res.txt')
