#!/usr/bin/env python3
"""0-pca.py"""


import numpy as np


def pca(X, var=0.95):
    """function that performs principal components"""
    U, S, Vt = np.linalg.svd(X)
    sum_s = np.cumsum(S)
    sum_s = sum_s / sum_s[-1]
    r = np.min(np.where(sum_s >= var))
    V = Vt.T
    Vr = V[..., :r + 1]
    return Vr
