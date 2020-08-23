#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm"""
    a = alpha
    b1 = beta1
    b2 = beta2
    e = epsilon
    Vd = (b1 * v) + ((1 - b1) * grad)
    Sd = (b2 * s) + ((1 - b2) * grad * grad)
    Vd_ok = Vd / (1 - b1 ** t)
    Sd_ok = Sd / (1 - b2 ** t)
    w = var - a * (Vd_ok / ((Sd_ok ** (0.5)) + e))
    return (w, Vd, Sd)
