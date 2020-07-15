#!/usr/bin/env python3

def mat_mul(mat1, mat2):

    if len(mat1[0]) != len(mat2):
        return None
    mat2_trans = [[row[idx] for row in mat2] for idx in range(len(mat2[0]))]
    return [[sum([a * b for a, b in zip(row1, row2)]) for row2 in mat2_trans]
        for row1 in mat1]
