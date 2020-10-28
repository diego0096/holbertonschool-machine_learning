#!/usr/bin/env python3
"""PCA: principal components analysis"""


import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    s = np.linalg.svd(X)[1]
    vh = np.linalg.svd(X)[2]
    num = np.cumsum(s)
    denom = np.sum(s)
    accum_var = num / denom
    num_truncated_results = np.argwhere(accum_var >= var)
    num_truncated_results = num_truncated_results[0, 0] + 1
    weights = vh[: num_truncated_results].T
    return weights
