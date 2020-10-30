#!/usr/bin/env python3
"""P affinities"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """that calculates the symmetric P affinities of a data set"""
    n = X.shape[0]
    D, P, beta, H = P_init(X, perplexity)
    for iter in range(n):
        beta_min, beta_max = None, None
        Di = D[iter, np.concatenate((np.r_[0:iter], np.r_[iter+1:n]))]
        Hi, Pi = HP(Di, beta[iter])
        Hdiff = Hi - H
        while abs(Hdiff) > tol:
            if Hdiff > 0:
                beta_min = beta[iter, 0]
                if beta_max is None:
                    beta[iter] = beta[iter] * 2
                else:
                    beta[iter] = (beta[iter] + beta_max) / 2
            else:
                beta_max = beta[iter, 0]
                if beta_min is None:
                    beta[iter] = beta[iter] / 2
                else:
                    beta[iter] = (beta[iter] + beta_min) / 2
            Hi, Pi = HP(Di, beta[iter])
            Hdiff = Hi - H
        P[iter, np.concatenate((np.r_[0:iter], np.r_[iter+1:n]))] = Pi
    P = (P + P.T) / (2 * n)
    return P
