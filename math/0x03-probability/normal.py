#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize normal distribution stats"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum([(x - self.mean) ** 2 for x in data]) /
                                (len(data))) ** .5)

    def z_score(self, x):
        """Calculate z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculate x score"""
        return z * self.stddev + self.mean
