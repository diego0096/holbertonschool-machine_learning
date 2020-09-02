#!/usr/bin/env python3
"""Module used to"""


import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """Function that trains a model"""
    history = network.fit(data,
                          labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          verbose=verbose)
    return(history)
