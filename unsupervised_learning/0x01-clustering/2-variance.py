#!/usr/bin/env python3
"""2-variance.py"""


import numpy as np


def variance(X, C):
    """function that calculates the total intra-cluster"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    n, d = X.shape
    k = C.shape[0]
    if k > n:
        return None
    if C.shape[1] != d:
        return None
    Xv = np.repeat(X, k, axis=0)
    Xv = Xv.reshape(n, k, d)
    Cv = np.tile(C, (n, 1))
    Cv = Cv.reshape(n, k, d)
    dist = np.linalg.norm(Xv - Cv, axis=2)
    short_dist = np.min(dist ** 2, axis=1)
    var = np.sum(short_dist)
    return var
