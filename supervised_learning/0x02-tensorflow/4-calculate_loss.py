#!/usr/bin/env python3
"""Calculate the loss of our network"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the loss of our network"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
