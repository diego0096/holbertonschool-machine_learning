#!/usr/bin/env python3
"""Transpose the matrix"""


def matrix_transpose(matrix):
    """Transpose a matrix"""
    return [[row[idx] for row in matrix] for idx in range(len(matrix[0]))]
