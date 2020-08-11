#!/usr/bin/env python3
"""Decode a one hot matrix to a numeric vector"""


import numpy as np


def one_hot_decode(one_hot):
    """Decode a one hot matrix to a numeric vector"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
