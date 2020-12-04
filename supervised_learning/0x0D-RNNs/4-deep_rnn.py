#!/usr/bin/env python3
"""4-deep_rnn.py"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """function that performs forward propagation for a deep RNN"""
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[2]
    k = len(rnn_cells)
    H = np.zeros((t + 1, k, m, h))
    Y = np.zeros((t, m, rnn_cells[k - 1].Wy.shape[1]))
    for i in range(t):
        for j in range(k):
            if i == 0:
                H[i, j] = h_0[j]
            if j == 0:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])
            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j],
                                                         H[i + 1, j - 1])
    return H, Y
