#!/usr/bin/env python
from typing import List

import numpy as np
from random import randint

if __name__ == '__main__':
    X = np.random.normal(loc=1, scale=10, size=(1000, 50))
    print(X)
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = ((X - m) / std)
    print(X_norm)

    values = []  # type: List[List[int]]
    for i in range(10):
        values.append([])
        for j in range(3):
            values[-1:][0].append(randint(0, 9))

    Z = np.array(values)
    print(Z)
    r = np.sum(Z, axis=1)
    print(np.nonzero(r > 10))

    A = np.eye(3)
    B = np.eye(3)
    print(A, '\n', B, '\n---')
    print(np.stack((A, B)))
    print(np.vstack((A, B)))
