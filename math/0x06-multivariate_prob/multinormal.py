#!/usr/bin/env python3
"""Multivariate normal distribution"""


import numpy as np


class MultiNormal:
    """Multivariate normal distribution"""
    def __init__(self, data):
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.ndarray((data.shape[0], data.shape[0]))
        for x in range(data.shape[0]):
            for y in range(data.shape[0]):
                self.cov[x][y] = (((data[x] - self.mean[x]) *
                                   (data[y] - self.mean[y])).sum() /
                                  (data.shape[1] - 1))
