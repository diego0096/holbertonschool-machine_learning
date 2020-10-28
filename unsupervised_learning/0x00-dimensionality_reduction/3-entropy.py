#!/usr/bin/env python3
"""3-entropy.py"""


import numpy as np


def HP(Di, beta):
    """function that calculates the Shannon entropy"""
    A = np.exp(-Di * beta)
    B = np.sum(np.exp(-Di * beta), axis=0)
    Pi = A / B
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
