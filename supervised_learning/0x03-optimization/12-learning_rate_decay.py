#!/usr/bin/env python3
"""Module used to"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay"""
    a = alpha
    a1 = tf.train.inverse_time_decay(learning_rate=a,
                                     global_step=global_step,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate,
                                     staircase=True)
    return a1
