#!/usr/bin/env python3
"""Simple deep neural network class"""


import numpy as np
import pickle


class DeepNeuralNetwork:
    """Simple deep neural network class"""

    def __init__(self, nx, layers):
        """nx: number of input features"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {"W1": np.random.randn(layers[0], nx) *
                          np.sqrt(2 / nx),
                          "b1": np.zeros((layers[0], 1))}
        for layer, size in enumerate(layers[1:], 2):
            cur = "W" + str(layer)
            self.__weights[cur] = (np.random.randn(size, layers[layer - 2]) *
                                   np.sqrt(2 / layers[layer - 2]))
            cur = "b" + str(layer)
            self.__weights[cur] = np.zeros((layers[layer - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward pragate the neural network"""
        self.__cache["A0"] = X
        for layer in range(self.__L):
            curw = "W" + str(layer + 1)
            curb = "b" + str(layer + 1)
            cura = "A" + str(layer + 1)
            preva = "A" + str(layer)
            z = (np.dot(self.__weights[curw], self.__cache[preva]) +
                 self.__weights[curb])
            self.__cache[cura] = 1 / (1 + np.exp(-z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate cost of neural network"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """Evaluate the neural network"""
        A = self.forward_prop(X)[0]
        return A.round().astype(int), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform a step of gradient descent on the network"""
        dz = {self.__L: cache["A" + str(self.__L)] - Y}
        Wstr = "W" + str(self.__L)
        for layer in range(self.__L - 1, 0, -1):
            curact = cache["A" + str(layer)]
            dz[layer] = (np.dot(self.__weights[Wstr].T, dz[layer + 1]) *
                         curact * (1 - curact))
            Wstr = "W" + str(layer)
        for layer in range(self.__L, 0, -1):
            Wstr = "W" + str(layer)
            bstr = "b" + str(layer)
            prevact = self.cache["A" + str(layer - 1)]
            self.__weights[Wstr] -= (np.matmul(dz[layer], prevact.T)
                                     * alpha / prevact.shape[1])
            self.__weights[bstr] -= (dz[layer].mean(axis=1, keepdims=True)
                                     * alpha)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1:
                raise ValueError("step must be a positive integer")
        itrcount = 0
        losses = []
        graphx = []
        while itrcount < iterations:
            if verbose and not (itrcount % step):
                print("Cost after {} iterations: {}"
                      .format(itrcount, self.cost(Y, self.forward_prop(X)[0])))
            if graph and not (itrcount % step):
                losses.append(self.cost(Y, self.__cache["A" + str(self.__L)]))
                graphx.append(itrcount)
            self.gradient_descent(Y, self.forward_prop(X)[1], alpha)
            itrcount += 1
        if verbose:
            self.forward_prop(X)
            print("Cost after {} iterations: {}"
                  .format(itrcount, self.cost(Y, self.__cache["A" +
                                                              str(self.__L)])))
        if graph:
            if itrcount % step:
                losses.append(self.cost(Y, self.cache["A" +
                                                      str(self.__L)]))
                graphx.append(itrcount)
            plt.plot(graphx, losses, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save class to .pkl"""
        if len(filename) < 4 or filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        """Load a DeepNeuralNetwork from file"""
        try:
            with open(filename, "rb") as infile:
                return pickle.load(infile)
        except FileNotFoundError:
            return None
