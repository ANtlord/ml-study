#!/usr/bin/python2
import os
import numpy as np
PROJECT_DIR = os.curdir
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def read_file(filename):
    """
    :type filename: str
    :rtype: tuple
    """
    pars = []
    class_ids = []
    with open(filename) as f:
        for line in f:
            arr = line.split()
            pars.append([int(x) for x in arr[:-1]])
            class_ids.append(int(arr[-1:][0]))

    return pars, class_ids


def classify(classifier):
    """
    :type classifier: native_bayes.ClassifierMixin
    :rtype: float:
    """
    pars, class_ids = read_file(os.path.join('shuttle', 'shuttle.tst.txt'))
    X = np.array(pars)
    Y = np.array(class_ids)
    classifier.fit(X, Y)

    pars, class_ids = read_file(os.path.join('shuttle', 'shuttle.trn'))
    predicted_classes = classifier.predict(pars)

    misstakes = 0
    for p_class, r_class in zip(predicted_classes, class_ids):
        if p_class != r_class:
            misstakes += 1
    return float(misstakes)/len(predicted_classes)*100


def main():
    print classify(GaussianNB())
    print classify(tree.DecisionTreeClassifier())
    print classify(svm.SVC(max_iter=5))
    print classify(KNeighborsClassifier(n_neighbors=1))


if __name__ == '__main__':
    main()
