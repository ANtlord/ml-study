#!/usr/bin/env python

from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np

from titanic import answer

if __name__ == '__main__':
    newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
    )
    y = newsgroups.target
    vectorizer = TfidfVectorizer()
    tf_idf_features = vectorizer.fit_transform(newsgroups.data)
    feature_mapping = vectorizer.get_feature_names()

    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(y.size, n_folds=5, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(tf_idf_features, y)

    parameter_C = max(gs.grid_scores_, key=lambda x: x.mean_validation_score).parameters['C']
    print(parameter_C)
    new_clf = SVC(parameter_C, kernel='linear', random_state=241)
    new_clf = new_clf.fit(tf_idf_features, y)
    weights = sorted(zip(new_clf.coef_.indices, new_clf.coef_.data), key=lambda x: abs(x[1]),
                     reverse=True)[:10]
    print(weights)
    word_indexes = ([x for x, y in weights])
    valueable_words = [feature_mapping[x] for x in word_indexes]
    valueable_words = sorted(valueable_words, key=str.lower)
    answer(' '.join(valueable_words), 'text_analyze_response.txt')
