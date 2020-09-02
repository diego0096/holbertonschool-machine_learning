#!/usr/bin/env python3
"""Module used to"""


import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                verbose=True,
                shuffle=False):
    """Function that trains a model using mini-batch"""
    def learning_rate(epochs):
        """ updates the learning rate using inverse time decay """
    return alpha / (1 + decay_rate * epochs)
    callbacks = []
    if (validation_data):
        early_stopping = K.callbacks.LearningRateScheduler(learning_rate, 1)

        callbacks.append(early_stopping)
    if (early_stopping and validation_data):
        stop_learn = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(stop_learn)
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return (history)
