#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1.copy() + row2.copy() for row1, row2 in zip(mat1, mat2)]
