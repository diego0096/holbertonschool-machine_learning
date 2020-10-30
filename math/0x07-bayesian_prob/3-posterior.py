#!/usr/bin/env python3
"""1-intersection.py"""


import numpy as np


def likelihood(x, n, P):
    """function that calculates the likelihood"""

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not all(P >= 0) or not all(P <= 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    A = (P ** x) * ((1 - P) ** (n - x))
    B = np.math.factorial(x) * np.math.factorial(n - x) / np.math.factorial(n)
    L = A / B
    return L


def intersection(x, n, P, Pr):
    """function that calculates the intersection"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(P >= 0) or not all(P <= 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not all(Pr >= 0) or not all(Pr <= 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1])[0]:
        raise ValueError("Pr must sum to 1")
    L = likelihood(x, n, P)
    IN = L * Pr
    return IN


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining the data"""
    IN = intersection(x, n, P, Pr)
    M = np.sum(IN)
    return M


def posterior(x, n, P, Pr):
    """calculates the posterior probability"""
    IN = intersection(x, n, P, Pr)
    M = marginal(x, n, P, Pr)
    P = IN / M
    return P
