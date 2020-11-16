#!/usr/bin/env python3
"""6-baum_welch.py"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """function that performs the forward algorithm"""
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """function that performs the Baum-Welch algorithm"""
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
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None
    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    N = Initial.shape[0]
    T = Observations.shape[0]
    M = Emission.shape[1]
    a = Transition
    b = Emission
    a_prev = np.copy(a)
    b_prev = np.copy(b)
    for iteration in range(1000):
        PF, F = forward(Observations, b, a, Initial)
        PB, B = backward(Observations, b, a, Initial)
        X = np.zeros((N, N, T - 1))
        NUM = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    Fit = F[i, t]
                    aij = a[i, j]
                    bjt1 = b[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1
        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN
        G = np.zeros((N, T))
        NUM = np.zeros((N, T))
        for t in range(T):
            for i in range(N):
                Fit = F[i, t]
                Bit = B[i, t]
                NUM[i, t] = Fit * Bit
        DEN = np.sum(NUM, axis=0)
        G = NUM / DEN
        a = np.sum(X, axis=2) / np.sum(G[:, :T - 1], axis=1)[..., np.newaxis]
        DEN = np.sum(G, axis=1)
        NUM = np.zeros((N, M))
        for k in range(M):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        b = NUM / DEN[..., np.newaxis]
        if np.all(np.isclose(a, a_prev)) or np.all(np.isclose(a, a_prev)):
            return a, b
        a_prev = np.copy(a)
        b_prev = np.copy(b)
    return a, b
