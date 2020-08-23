#!/usr/bin/env python3
"""Module used to"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm"""
    a = alpha
    b2 = beta2
    e = epsilon
    a = alpha
    dw = grad
    w = var
    s_new = b2 * s + (1 - b2) * (dw * dw)
    W = w - a * (dw / ((s_new ** 0.5) + e))
    return W, s_new
