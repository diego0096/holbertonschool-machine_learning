#!/usr/bin/env python3
"""1-pca.py"""


import numpy as np


def pca(X, ndim):
    """function that performs principal components"""
    X = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X)
    Tr = np.matmul(U[..., :ndim], np.diag(S[..., :ndim]))
    return Tr
