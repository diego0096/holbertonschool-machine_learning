#!/usr/bin/env python3
"""1-gp.py"""


import numpy as np


class GaussianProcess:
    """Class that instantiates a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """define and initialize variables and methods"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """function that calculates the covariance kernel matrix"""
        a = np.sum(X1 ** 2, axis=1, keepdims=True)
        b = np.sum(X2 ** 2, axis=1, keepdims=True)
        c = np.matmul(X1, X2.T)
        dist_sq = a + b.reshape(1, -1) - 2 * c
        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)
        return K

    def predict(self, X_s):
        """function that predicts the mean and standard deviation"""
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        Y = self.Y
        K_inv = np.linalg.inv(K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), Y).reshape(-1)
        cov_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        sigma = np.diag(cov_s)
        return mu_s, sigma

def update(self, X_new, Y_new):
        """function that updates a Gaussian process"""
        self.X = np.concatenate((self.X, X_new[..., np.newaxis]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[..., np.newaxis]), axis=0)
        self.K = self.kernel(self.X, self.X)
