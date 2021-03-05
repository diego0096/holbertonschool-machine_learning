#!/usr/bin/env python3
"""Compute policy output and updates"""


import numpy as np


def policy(state, weight):
    """Compute policy output"""
    res = state @ weight
    res = np.exp(res)
    return res / res.sum()
