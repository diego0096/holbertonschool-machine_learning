#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""


import matplotlib.pyplot as plt
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
        """getter method        """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights["W" + str(layer + 1)]
            a_ = self.__cache["A" + str(layer)]
            b = self.__weights["b" + str(layer + 1)]
            z = np.matmul(weights, a_) + b
            forward_prop = 1 / (1 + np.exp(-1 * z))
            self.__cache["A" + str(layer + 1)] = forward_prop
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        j = - (1 / m)
        Â = 1.0000001 - A
        Ŷ = 1 - Y
        log_A = np.log(A)
        log_Â = np.log(Â)
        cost = j * np.sum(np.multiply(Y, log_A) + np.multiply(Ŷ, log_Â))
        return cost

    def evaluate(self, X, Y):
        """Calculates the cost of the model using logistic regression"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__cache["A" + str(self.L)])
        labels = np.where(self.__cache["A" + str(self.L)] < 0.5, 0, 1)
        return (labels, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        cp_w = self.__weights.copy()
        la = self.__L
        dz = self.__cache['A' + str(la)] - Y
        dw = np.dot(self.__cache['A' + str(la - 1)], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        self.__weights['W' + str(la)] = cp_w['W' + str(la)] - alpha * dw.T
        self.__weights['b' + str(la)] = cp_w['b' + str(la)] - alpha * db
        for la in range(self.__L - 1, 0, -1):
            g = self.__cache['A' + str(la)] * (1 - self.__cache['A' + str(la)])
            dz = np.dot(cp_w['W' + str(la + 1)].T, dz) * g
            dw = np.dot(self.__cache['A' + str(la - 1)], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W' + str(la)] = cp_w['W' + str(la)] - alpha * dw.T
            self.__weights['b' + str(la)] = cp_w['b' + str(la)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network and Updates the private
        attributes __weights and __cache"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if (iterations < 0):
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if (alpha < 0):
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if (step < 1) or (step > iterations):
                raise ValueError('step must be positive and <= iterations')
        steps = []
        costs = []
        for cont in range(iterations + 1):
            self.forward_prop(X)
            cache = self.__cache
            self.gradient_descent(Y, cache, alpha)
            if cont == iterations or cont % step == 0:
                cost = self.cost(Y, self.__cache['A' + str(self.__L)])
                if verbose:
                    print('Cost after {} iterations: {}'.format(cont, cost))
                if graph:
                    costs.append(cost)
                    steps.append(cont)
        if graph:
            fig = plt.figure(figsize=(10, 10))
            plt.plot(steps, costs, linewidth=3, markevery=10)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            fig.set_facecolor("white")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        filename = filename + ".pkl" if ('.pkl'not in filename) else filename
        try:
            file_binary = open(filename, 'wb')
            pickle.dump(self, file_binary)
            file_binary.close()
        except BaseException:
            return None

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            DeepNeuralNetwork_file = open(filename, 'rb')
            info = pickle.load(DeepNeuralNetwork_file)
            DeepNeuralNetwork_file.close()
            return info
        except BaseException:
            return None
