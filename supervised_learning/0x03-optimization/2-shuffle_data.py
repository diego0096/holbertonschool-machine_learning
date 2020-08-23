#!/usr/bin/env python3
"""Shuffle data in two matrices"""


import numpy as np


def shuffle_data(X, Y):
    """Shuffle data in two matrices"""
    shufflidx = np.random.permutation(X.shape[0])
    return X[shufflidx], Y[shufflidx]
