#!/usr/bin/env python3
"""0-initialize.py"""


import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None
    mu = np.random.uniform(low=np.min(X, axis=0),
                           high=np.max(X, axis=0),
                           size=(k, d))
    return mu
