#!/usr/bin/env python
from sklearn.svm import SVC
from titanic import answer
import numpy as np

if __name__ == '__main__':
    train_data = np.genfromtxt('svm-data.csv', delimiter=',')
    X_train_data = features = train_data[:, 1:]
    Y_train_data = train_data[:, 0]
    clf = SVC(random_state=241, C=100000, kernel='linear')
    clf = clf.fit(X_train_data, Y_train_data)
    answer(' '.join([str(x+1) for x in clf.support_]), 'svm_learn_response.txt')
