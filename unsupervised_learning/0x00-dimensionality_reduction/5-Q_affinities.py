#!/usr/bin/env python3
"""Q affinities"""


import numpy as np


def Q_affinities(Y):
    """that calculates the Q affinities"""
    n = Y.shape[0]
    Q = np.zeros((n, n))
    sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
    num = sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T)
    num = (1 + num) ** -1
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)
    return(Q, num)
