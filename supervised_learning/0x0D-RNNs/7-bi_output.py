#!/usr/bin/env python3
"""7-bi_output.py"""


import numpy as np


class BidirectionalCell:
    """define the class BidirectionalCell"""

    def __init__(self, i, h, o):
        """constructor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation"""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """function that performs backward propagation"""
        cell_input = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(cell_input, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """function that calculates all outputs for the RNN"""
        t = H.shape[0]
        m = H.shape[1]
        Y = np.zeros((t, m, self.Wy.shape[1]))
        for i in range(t):
            Y[i] = self.softmax(np.matmul(H[i], self.Wy) + self.by)
        return Y

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))
