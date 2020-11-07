#!/usr/bin/env python3
"""1-kmeans.py"""


import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None
    C = np.random.uniform(low=np.min(X, axis=0),
                          high=np.max(X, axis=0),
                          size=(k, d))
    return C


def kmeans(X, k, iterations=1000):
    """function that performs K-means clustering on a dataset"""
    C = initialize(X, k)
    if C is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    for iteration in range(iterations):
        C_prev = np.copy(C)
        Xv = np.repeat(X, k, axis=0)
        Xv = Xv.reshape(n, k, d)
        Cv = np.tile(C, (n, 1))
        Cv = Cv.reshape(n, k, d)
        dist = np.linalg.norm(Xv - Cv, axis=2)
        clss = np.argmin(dist ** 2, axis=1)
        for j in range(k):
            indices = np.where(clss == j)[0]
            if len(indices) == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[indices], axis=0)
        if (C == C_prev).all():
            return C, clss
    Cv = np.tile(C, (n, 1))
    Cv = Cv.reshape(n, k, d)
    dist = np.linalg.norm(Xv - Cv, axis=2)
    clss = np.argmin(dist ** 2, axis=1)
    return C, clss
