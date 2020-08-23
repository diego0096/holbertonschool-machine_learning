#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent"""
    α = alpha
    β1 = beta1
    dw = grad
    w = var
    v_dw = β1 * v + (1 - β1) * dw
    W = w - (α * v_dw)
    return W, v_dw
