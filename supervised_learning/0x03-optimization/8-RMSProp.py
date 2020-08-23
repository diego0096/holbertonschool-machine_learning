#!/usr/bin/env python3
"""Module used to"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Updates a variable using the RMSProp optimization algorithm"""

    a = alpha
    b2 = beta2
    e = epsilon
    train = tf.train.RMSPropOptimizer(learning_rate=a, decay=b2, epsilon=e)
    op = train.minimize(loss)
    return op
