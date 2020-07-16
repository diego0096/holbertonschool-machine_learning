#!usr/bin/env python3
"""numpy to concatenates two matrices"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenate matrices"""
    return np.append(mat1, mat2, axis=axis)
