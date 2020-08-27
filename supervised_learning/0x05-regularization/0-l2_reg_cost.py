#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization"""
    if (L == 0):
        return 0
    λ = lambtha
    sum_weights = 0
    for keys in weights:
        if (keys[0] == "W"):
            values = weights[keys]
            sum_weights += np.linalg.norm(values)
    cost_l2 = cost + (λ / (2 * m)) * sum_weights
    return(cost_l2)
