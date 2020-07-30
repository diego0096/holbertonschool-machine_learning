#!/usr/bin/env python3
"""Poisson calculations"""


class Poisson:
    """Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize distributions"""
        self.lambtha = float(lambtha)
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

    def pmf(self, k):
        """number of events"""
        if k < 0:
            return 0
        k = int(k)
        return (pow(self.lambtha, k) *
                pow(2.7182818285, -1 * self.lambtha) /
                factorial(k))

    def cdf(self, k):
        """CDF at k events"""
        if k < 0:
            return 0
        k = int(k)
        return sum([self.pmf(n) for n in range(k + 1)])


def factorial(n):
    """return factorial"""
    if n < 0:
        return None
    if n == 0:
        return 1
    if n < 2:
        return 1
    return n * factorial(n-1)
