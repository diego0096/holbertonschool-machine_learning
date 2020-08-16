#!/usr/bin/env python3
"""Create Tensor placeholders"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """Create Tensor placeholders"""
    return (tf.placeholder(float, shape=[None, nx], name='x'),
            tf.placeholder(float, shape=[None, classes], name='y'))
