#!/usr/bin/env python3
"""Module used to"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Updates a variable using the RMSProp optimization algorithm"""

    α = alpha
    β2 = beta2
    ε = epsilon
    train = tf.train.RMSPropOptimizer(learning_rate=α, decay=β2, epsilon=ε)
    op = train.minimize(loss)
    return op
