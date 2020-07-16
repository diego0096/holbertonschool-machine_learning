#!usr/bin/env python3
"""transpose the matrix"""


def matrix_transpose(matrix):
    """transpose matrix"""
    return [[row[idx] for row in matrix] for idx in range(len(matrix[0]))]
