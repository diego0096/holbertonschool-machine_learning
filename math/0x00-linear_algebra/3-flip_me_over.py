#!usr/bin/env python3
"""transpose the matrix"""


def matrix_transpose(matrix):
    """transpose matrix"""

    return [[row[zero] for row in matrix] for zero in range(len(matrix[0]))]
