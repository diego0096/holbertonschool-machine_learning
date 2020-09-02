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
                verbose=True,
                shuffle=False):
    """Function that trains a model using mini-batch"""
    callbacks = []
    if (validation_data):
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return (history)
