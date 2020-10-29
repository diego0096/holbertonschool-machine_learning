#!/usr/bin/env python3
"""PCA: principal components analysis"""


import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset"""
    normal = np.mean(X, axis=0)
    X_normal = X - normal
    vh = np.linalg.svd(X_normal)[2]
    Weights_r = vh[: ndim].T
    T = np.matmul(X_normal, Weights_r)

    return T
