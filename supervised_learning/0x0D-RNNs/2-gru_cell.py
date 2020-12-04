#!/usr/bin/env python3
"""2-gru_cell.py"""


import numpy as np


class GRUCell:
    """define the class GRUCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        update_gate = self.sigmoid(np.matmul(cell_input, self.Wz) + self.bz)
        reset_gate = self.sigmoid(np.matmul(cell_input, self.Wr) + self.br)
        updated_cell_input = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        h_r = np.tanh(np.matmul(updated_cell_input, self.Wh) + self.bh)
        h_next = update_gate * h_r + (1 - update_gate) * h_prev
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-Y))
