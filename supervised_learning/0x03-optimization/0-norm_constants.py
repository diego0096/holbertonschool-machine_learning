#!/usr/bin/env python3
"""Calculate normalization"""


import numpy as np


def normalization_constants(X):
    """Calculate normalization"""
    return X.mean(axis=0), X.std(axis=0)
