#!/usr/bin/env python3
"""Create a tensorflow layer"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a tensorflow layer"""
    initializer = (tf.contrib.layers.
                   variance_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initializer)(prev)
