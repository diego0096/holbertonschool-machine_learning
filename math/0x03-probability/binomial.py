#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """init binominal"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            self.p = -1 * (variance / mean - 1)
            n = mean/self.p
            self.n = round(n)
            self.p *= n/self.n

    def pmf(self, k):
        """return probaility"""
        k = int(k)
        if k > self.n or k < 0:
            return 0
        return (factorial(self.n) / factorial(k) / factorial(self.n - k) *
                self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """return probability of 0 to k"""
        if k > self.n or k < 0:
            return 0
        cdf_sum = 0
        for x in range (0, int(k) + 1):
            cdf_sum += self.pmf(x)
        return cdf_sum


def factorial(x):
    """return factorial"""
    if x < 0:
        return None
    if x < 2:
        return 1
    return x * factorial(x - 1)
