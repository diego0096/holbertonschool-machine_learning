#!/usr/bin/env python3
"""Create a Layer with L2 Dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """function that creates a tf layer using dropout"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    drop = tf.layers.Dropout(rate=(1 - keep_prob))
    return drop(layer(prev))
