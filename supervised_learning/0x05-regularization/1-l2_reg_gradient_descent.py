#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """function that updates the weights and biases"""
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        m = Y.shape[1]
        if i != L:
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), 1 - cache['A' + str(i)] ** 2)
        else:
            dZi = cache['A' + str(i)] - Y
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m
        l2 = (1 - alpha * lambtha / m)
        weights['W' + str(i)] = l2 * weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * dbi
