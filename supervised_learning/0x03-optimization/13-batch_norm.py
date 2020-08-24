#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a NN"""
    β = beta
    γ = gamma
    ε = epsilon
    μ = Z.mean(0)
    σ = Z.std(0)
    σ2 = Z.std(0) ** 2
    z_normalized = (Z - μ) / ((σ2 + ε) ** (0.5))
    Ẑ = γ * z_normalized + β
    return Ẑ
