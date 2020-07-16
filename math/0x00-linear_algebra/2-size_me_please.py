#!/usr/bin/env python3


def matrix_shape(matrix):
    shape = []
    try:
        while(len(matrix) > 0):
            shape.append(len(matrix))
            matrix = matrix[0]
    except TypeError:
        pass
    return shape
