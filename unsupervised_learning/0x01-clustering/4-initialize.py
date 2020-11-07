#!/usr/bin/env python3
"""4-initialize.py"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function that initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None
    pi = np.full(shape=(k,), fill_value=1/k)
    m, _ = kmeans(X, k)
    Sij = np.diag(np.ones(d))
    S = np.tile(Sij, (k, 1)).reshape(k, d, d)
    return pi, m, S
