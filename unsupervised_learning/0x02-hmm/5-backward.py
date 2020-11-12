#!/usr/bin/env python3
"""5-backward.py"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """function that performs the backward algorithm"""
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1),
                      np.ones(Initial.shape[0])).all():
        return None, None
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None
    N = Initial.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))
    for j in range(T - 2, -1, -1):
        for i in range(N):
            B[i, j] = np.sum(B[:, j + 1] * Emission[:, Observation[j + 1]]
                             * Transition[i, :], axis=0)
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]
    return P, B
