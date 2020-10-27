#!/usr/bin/env python3

"""Class for the Multinirmal probabilities"""


import numpy as np


class MultiNormal:
    """ Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if(isinstance(data, type(None))):
            raise TypeError('data must be a 2D numpy.ndarray')
        if (not isinstance(data, np.ndarray)) or (len(data.shape)) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if (data.shape[1] < 2):
            raise ValueError("data must contain multiple data points")
        data = data.T
        mean = data.mean(axis=0)
        mean = np.reshape(mean, (-1, data.shape[1]))
        n = data.shape[0] - 1
        x = data - mean
        cov = np.dot(x.T, x) / n
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """Method that calculates the PDF at a data point"""
        d = self.cov.shape[0]
        if(isinstance(x, type(None))):
            raise TypeError('x must be a numpy.ndarray')
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if (x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        if (len(x.shape) != 2) or (x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        mean = self.mean
        cov = self.cov
        x_m = x - mean
        pdf = (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(cov)))
               * np.exp(-(np.linalg.solve(cov, x_m).T.dot(x_m)) / 2))
        return pdf[0][0]
