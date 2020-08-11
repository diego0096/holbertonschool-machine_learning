#!/usr/bin/env python3
"""Convert a numeric label vector"""


import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector"""
    try:
        onehot = np.zeros((classes, Y.shape[0]))
        for ex, label in enumerate(Y):
            onehot[label][ex] = 1
        return onehot
    except Exception:
        return None
