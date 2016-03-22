#!/usr/bin/env python
import operator

from pandas import DataFrame
import pandas
from sklearn.tree import DecisionTreeClassifier
TARGET_COLNAME = 'Survived'

if __name__ == '__main__':
    X_cols = ['Pclass', 'Fare', 'Age', 'Sex']
    data = pandas.read_csv('train.csv', usecols=X_cols+[TARGET_COLNAME])  # type: DataFrame

    filtered_features = data[pandas.notnull(data['Age'])]
    X = pandas.concat([filtered_features[x] for x in X_cols], axis=1)  # type: DataFrame
    X['Sex'] = X['Sex'].replace('male', 0).replace('female', 1)
    Y = filtered_features[TARGET_COLNAME]
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, Y)
    top_importances = sorted(enumerate(clf.feature_importances_), key=operator.itemgetter(1),
                             reverse=True)[:2]

    with open('titanic_trees.txt', 'w') as f:
        f.write(' '.join([X_cols[k] for k, v in top_importances]))
