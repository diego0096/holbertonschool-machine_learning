#!/usr/bin/env python3
"""Advanced Linear Algebra"""


def determinant(matrix):
    """function that calculates the determinant of a matrix"""

    err_1 = "matrix must be a list of lists"
    if not isinstance(matrix, list):
        raise TypeError(err_1)
    if not all([isinstance(element, list) for element in matrix]):
        raise TypeError(err_1)
    if len(matrix) == 0:
        raise TypeError(err_1)
    height = len(matrix)
    width = len(matrix[0])
    if height == 1 and width == 0:
        return 1

    err_2 = "matrix must be a square matrix"
    if height != width:
        raise ValueError(err_2)
    if not all([len(matrix[i]) == width for i in range(1, height)]):
        raise ValueError(err_2)
    if height == 1 and width == 1:
        return matrix[0][0]
    if height == 2 and width == 2:
        sub_det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return sub_det

    det = 0
    for col_index in range(width):
        submat = matrix[1:]
        new_height = len(submat)
        for row_index in range(new_height):
            submat[row_index] = (submat[row_index][0: col_index] +
                                 submat[row_index][col_index + 1:])
        sign = (-1) ** (col_index % 2)
        sub_det = determinant(submat)
        det += sign * matrix[0][col_index] * sub_det

    return det
