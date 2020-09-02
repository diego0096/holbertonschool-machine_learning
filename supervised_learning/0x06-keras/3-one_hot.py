#!/usr/bin/env python3
"""Convert a numeric label vector"""


import tensorflow.keras as K


def one_hot(Y, classes=None):
    """Convert a numeric label vector"""
    if classes is None:
        classes = max(Y) + 1
    return K.utils.to_categorical(Y, classes)
