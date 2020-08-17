#!/usr/bin/env python3
"""Create training op for our network"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Create training op for our network"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
