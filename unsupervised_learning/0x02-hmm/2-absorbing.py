#!/usr/bin/env python3
"""1-regular.py"""


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


def regular(P):
    """function that determines the steady state probabilities"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    s = np.full(n, (1 / n))[np.newaxis, ...]
    Pk = np.copy(P)
    s_prev = s
    while True:
        Pk = np.matmul(Pk, P)
        if np.any(Pk <= 0):
            return None
        s = np.matmul(s, P)
        if np.all(s_prev == s):
            return s
        s_prev = s


def absorbing(P):
    """function that determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    if np.all(np.diag(P) != 1):
        return False
    if np.all(np.diag(P) == 1):
        return True
    for i in range(n):
        if np.any(P[i, :] == 1):
            continue
        break
    II = P[:i, :i]
    Id = np.identity(n - i)
    R = P[i:, :i]
    Q = P[i:, i:]
    try:
        F = np.linalg.inv(Id - Q)
    except Exception:
        return False
    FR = np.matmul(F, R)
    Pbar = np.zeros((n, n))
    Pbar[:i, :i] = P[:i, :i]
    Pbar[i:, :i] = FR
    Qbar = Pbar[i:, i:]
    if np.all(Qbar == 0):
        return True
