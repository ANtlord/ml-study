#!/usr/bin/python2
import os
import numpy as np
from sklearn import linear_model
PROJECT_DIR = os.curdir
import csv


def read_data():
    res = []
    with open(os.path.join(PROJECT_DIR, 'winequality-red.csv')) as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            res.append(line)
    return res


def parse_data(data):
    """
    :type data: list
    """

    def get_parameters(data):
        """
        :type data: list
        :rtype: list
        """
        res = []
        for row in data[1-len(data):]:
            res.append([float(x) for x in row[:-1]])
        return res

    def get_results(data):
        """
        :type data: list
        :rtype: list
        """
        res = []
        for row in data[1-len(data):]:
            res.append(float(row[-1:][0]))
        return res

    parameters = get_parameters(data)
    results = get_results(data)

    return parameters, results


def get_linear_approximation(x, y):
    """
    :type x: list
    :type y: list
    :rtype: tuple
    """
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    predicts = regr.predict(x)

    abs_delta = 0
    for a, b in zip(predicts, y):
        abs_delta += abs(a-b)

    abs_delta = abs_delta / len(y)
    return abs_delta, np.average(y), abs_delta/np.average(y)*100


def get_ridge_approximation(x, y):
    """
    :type x: list
    :type y: list
    :rtype: tuple
    """
    regr = linear_model.Ridge()
    regr.fit(x, y)

    predicts = regr.predict(x)

    abs_delta = 0
    for a, b in zip(predicts, y):
        abs_delta += abs(a-b)

    abs_delta = abs_delta / len(y)
    return abs_delta, np.average(y), abs_delta/np.average(y)*100


def main():
    data = read_data()
    x, y = parse_data(data)

    abs_delta, average, rel_delta = get_ridge_approximation(x, y)

    print abs_delta
    print average
    print rel_delta


if __name__ == '__main__':
    main()
