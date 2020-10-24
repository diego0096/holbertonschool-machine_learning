#!/usr/bin/env python3
"""Advanced Linear Algebra"""
import numpy as np


def definiteness(matrix):
    """function that determines the definiteness of a matrix"""

    err_1 = "matrix must be a numpy.ndarray"
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err_1)
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.array_equal(matrix, matrix.T):
        return None
    height = matrix.shape[0]
    width = matrix.shape[1]
    D = []
    for row_index, col_index in zip(range(height), range(width)):
        det = np.linalg.det(matrix[:row_index + 1, :col_index + 1])
        D.append(det)
    D = np.array(D)
    if all(D > 0):
        return "Positive definite"

    if all(D[0::2] < 0) and all(D[1::2] > 0):
        return "Negative definite"

    if D[-1] != 0:
        return "Indefinite"

    if D[-1] == 0 and all(D[:-1] > 0):
        return "Positive semi-definite"

    if D[-1] == 0 and all(D[0:-1:2] < 0) and all(D[1:-1:2] > 0):
        return "Negative semi-definite"
