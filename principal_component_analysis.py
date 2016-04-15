#!/usr/bin/env python
import pandas
from pandas import DataFrame
from titanic import answer
from sklearn.decomposition import PCA
import numpy as np

if __name__ == '__main__':
    data_train = pandas.read_csv('close_prices.csv', index_col=None)  # type: DataFrame
    data_indexes = pandas.read_csv('djia_index.csv', index_col=None)  # type: DataFrame
    X_train = data_train.values[:, 1:]
    pca = None
    i = None
    for i in range(1, X_train.shape[1]):
        pca = PCA(i)
        pca.fit(X_train)
        print(i, pca.explained_variance_ratio_)
        if sum(pca.explained_variance_ratio_) > 0.9:
            break

    answer(i, 'pca_1.txt')
    transformed_features = pca.transform(X_train)
    pearson_c = np.corrcoef([transformed_features[:, 0], data_indexes['^DJI']])[1, 0]
    answer(pearson_c, 'pca_2.txt')

    index = np.argmax(pca.components_[0])
    answer(data_train.keys()[1:][index], 'pca_3.txt')
