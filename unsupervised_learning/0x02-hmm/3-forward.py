#!/usr/bin/env python3
"""3-forward.py"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """function that performs the forward algorithm for a hidden markov model"""
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
    F = np.zeros((N, T))
    index = Observation[0]
    Emission_idx = Emission[:, index]
    F[:, 0] = Initial.T * Emission_idx
    for j in range(1, T):
        for i in range(N):
            F[i, j] = np.sum(Emission[i, Observation[j]]
                             * Transition[:, i] * F[:, j - 1], axis=0)
    P = np.sum(F[:, T-1:], axis=0)[0]
    return P, F
