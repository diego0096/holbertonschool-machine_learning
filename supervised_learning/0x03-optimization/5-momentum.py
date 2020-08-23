#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent"""
    a = alpha
    b1 = beta1
    dw = grad
    w = var
    v_dw = b1 * v + (1 - b1) * dw
    W = w - (a * v_dw)
    return W, v_dw
