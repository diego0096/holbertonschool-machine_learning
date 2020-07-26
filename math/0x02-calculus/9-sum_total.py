#!/usr/bin/env python3
"""Calculate sumation"""


def summation_i_squared(n):
    """Calculate sumation"""
    if type(n) is not int or n < 1:
        return None
    return int(n / 6 + n * n / 2 + pow(n, 3) / 3)
