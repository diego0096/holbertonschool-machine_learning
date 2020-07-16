#!/usr/bin/env python3
"""add two arrays elemnt 2D"""


def add_matrices2D(mat1, mat2):
    """two elements 2D"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[ele1 + ele2 for ele1, ele2 in zip(row1, row2)]
            for row1, row2 in zip(mat1, mat2)]
