#!/usr/bin/env python3
"""Module used to"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow"""

    α = alpha
    β1 = beta1
    train = tf.train.MomentumOptimizer(learning_rate=α, momentum=β1)
    op = train.minimize(loss)
    return op
