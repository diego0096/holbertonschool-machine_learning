#!/usr/bin/env python3
"""Normalize input data"""


import numpy as np


def normalize(X, m, s):
    """Normalize input data"""
    return (X - m) / s
