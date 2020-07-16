#!usr/bin/env python3


def matrix_transpose(matrix):

    return [[row[zero] for row in matrix] for zero in range(len(matrix[0]))]
