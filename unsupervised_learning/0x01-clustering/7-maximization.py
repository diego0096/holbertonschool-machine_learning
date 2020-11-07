#!/usr/bin/env python3
"""7-maximization.py"""


import numpy as np


def maximization(X, g):
    """function that calculates the maximization step"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), np.ones(n,)).all():
        return None, None, None
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        gn = np.sum(g[i], axis=0)
        pi[i] = gn / n
        m[i] = np.sum(np.matmul(g[i][np.newaxis, ...], X), axis=0) / gn
        S[i] = np.matmul(g[i][np.newaxis, ...] * (X - m[i]).T, (X - m[i])) / gn
    return pi, m, S
