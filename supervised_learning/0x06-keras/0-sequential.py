#!/usr/bin/env python3
"""Module used to"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a NN with the Keras library"""
    l = lambtha

    a_model = K.Sequential()
    n_layers = len(layers)
    regularizer = K.regularizers.l2(l)
    for i in range(n_layers):
        a_model.add(K.layers.Dense(
            units=layers[i],
            input_dim=nx,
            kernel_regularizer=regularizer,
            activation=activations[i],
        )
        )
        if i < n_layers - 1:
            a_model.add(K.layers.Dropout(1 - keep_prob))
    return a_model
