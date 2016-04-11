#!/usr/bin/env python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from titanic import answer

import pandas
from sklearn.linear_model import Ridge

if __name__ == '__main__':
    vectorizer = TfidfVectorizer(min_df=5)
    enc = DictVectorizer()
    data_train = pandas.read_csv('salary-train.csv', index_col=None)  # type: DataFrame
    data_test = pandas.read_csv('salary-test-mini.csv', index_col=None)  # type: DataFrame

    for key in data_train.keys()[:3]:
        data_train[key] = data_train[key].str.lower()
    data_train.replace('[^a-zA-Z0-9]', ' ', regex=True, inplace=True)
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)

    data_test['LocationNormalized'].fillna('nan', inplace=True)
    data_test['ContractTime'].fillna('nan', inplace=True)

    X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']]
                                      .to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']]
                                 .to_dict('records'))

    tf_idf_features = vectorizer.fit_transform(data_train['FullDescription'])
    tf_idf_features_test = vectorizer.transform(data_test['FullDescription'])

    train_features = hstack((tf_idf_features, X_train_categ), format='csr')
    test_features = hstack((tf_idf_features_test, X_test_categ), format='csr')
    regressor = Ridge(random_state=241, alpha=1)
    regressor.fit(train_features, data_train['SalaryNormalized'])
    res = regressor.predict(test_features)

    answer('%0.2f %0.2f' % (res[0], res[1]), 'salary_res.txt')
    answer('%s %s' % (res[0], res[1]), 'salary_res_2.txt')
