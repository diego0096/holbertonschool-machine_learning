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

    def pdf(self, x):
        """Return pdf of the multivariate normal distribution"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if ((len(x.shape) != 2 or x.shape[1] != 1
             or x.shape[0] != self.mean.shape[0])):
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.mean.shape[0]))
        res = np.matmul((x - self.mean).T, np.linalg.inv(self.cov))
        res = np.exp(np.matmul(res, (x - self.mean)) / -2)
        res /= np.sqrt(pow(2 * np.pi, x.shape[0]) * np.linalg.det(self.cov))
        return res[0][0]
