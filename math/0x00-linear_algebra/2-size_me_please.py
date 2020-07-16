#!/usr/bin/env python3
"""
Module used to find the size of a matrix
"""


def matrix_shape(matrix):
    shape = []
    try:
        while(len(matrix) > 0):
            shape.append(len(matrix))
            matrix = matrix[0]
    except TypeError:
        pass
    return shape
