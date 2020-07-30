#!/usr/bin/env python3
"""Poisson calculations"""


class Poisson:
    """Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize distributions"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambrha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
