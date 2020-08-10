#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""


import numpy as np


class DeepNeuralNetwork:
    """ Class """
    def __init__(self, nx, layers):
        """Initialize NeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(len(layers)):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError('layers must be a list of positive integers')

            self.weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))

            if lay == 0:
                sq = np.sqrt(2 / nx)
                he_et_al = np.random.randn(layers[lay], nx) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

            else:
                sq = np.sqrt(2 / layers[lay - 1])
                he_et_al = np.random.randn(layers[lay], layers[lay - 1]) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

    @property
    def L(self):
        """getter method"""
        return self.__L

    @property
    def cache(self):
        """getter method"""
        return self.__cache

    @property
    def weights(self):
        """getter method"""
        return self.__weights
