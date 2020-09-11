#!/usr/bin/env python3
"""Backpropagation over Pooling Layer"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs a backpropagation"""
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c = dA.shape[3]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    func = {'max': np.max, 'avg': np.mean}
    dA_prev = np.zeros(shape=A_prev.shape)
    if mode in ['max', 'avg']:
        for img_num in range(m):
            for k in range(c):
                for i in range(h_new):
                    for j in range(w_new):
                        window = A_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ]
                        if mode == 'max':
                            mask = np.where(window == np.max(window), 1, 0)
                        elif mode == 'avg':
                            mask = np.ones(shape=window.shape)
                            mask /= (kh * kw)
                        dA_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ] += mask * dA[
                            img_num,
                            i,
                            j,
                            k
                        ]
    return dA_prev
