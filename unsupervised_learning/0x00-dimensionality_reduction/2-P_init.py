#!/usr/bin/env python3
"""2-P_init.py"""


import numpy as np


def P_init(X, perplexity):
    """function that initializes variables"""
    n, d = X.shape
    D = (np.sum(X ** 2, axis=1) - 2 * np.matmul(X, X.T) +
         np.sum(X ** 2, axis=1)[..., np.newaxis])
    D[[range(n)], range(n)] = 0
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
