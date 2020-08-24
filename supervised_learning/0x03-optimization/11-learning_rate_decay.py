#!/usr/bin/env python3
"""Module used to"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy"""
    α = alpha
    dr = decay_rate
    decayed_learning_rate = α / (1 + dr * int(global_step / decay_step))
    return (decayed_learning_rate)
