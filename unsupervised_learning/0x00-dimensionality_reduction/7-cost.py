#!/usr/bin/env python3
"""Function Cost"""


import numpy as np


def cost(P, Q):
    """Function that calculates the cost"""
    Q = np.maximum(Q, 1e-12)
    P = np.maximum(P, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
