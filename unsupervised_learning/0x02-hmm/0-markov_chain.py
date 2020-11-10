#!/usr/bin/env python3
"""0-markov_chain.py"""


import numpy as np


def markov_chain(P, s, t=1):
    """function that determines the probability of a markov chain"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[1]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    if not np.isclose(np.sum(s, axis=0), [1])[0]:
        return None
    Pt = np.linalg.matrix_power(P, t)
    Ps = np.matmul(s, Pt)
    return Ps
