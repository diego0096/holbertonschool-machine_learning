#!/usr/bin/env python3
"""Calculate derivate"""


def poly_derivative(poly):
    """Function that calculates a derivate of a poly"""
    if not isinstance(poly, list):
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        deriv = [0] * (len(poly) - 1)
        for i in range(len(poly) - 1):
            if (not isinstance(poly[i], int)):
                return None
            deriv[i] = poly[i + 1] * (i + 1)
    return deriv
