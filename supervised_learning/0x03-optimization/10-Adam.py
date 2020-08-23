#!/usr/bin/env python3
"""Module used to"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates the training operation"""
    a = alpha
    b1 = beta1
    b2 = beta2
    e = epsilon
    train = tf.train.AdamOptimizer(learning_rate=a,
                                   beta1=b1,
                                   beta2=b2,
                                   epsilon=e)
    op = train.minimize(loss)
    return op
