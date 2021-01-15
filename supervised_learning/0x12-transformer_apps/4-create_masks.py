#!/usr/bin/env python3
"""4-create_masks.py"""


import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """function that creates all masks for training/validation"""
    inputs = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    target = tf.cast(tf.math.equal(target, 0), tf.float32)
    encoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    decoder_target_mask = target[:, tf.newaxis, tf.newaxis, :]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target.shape[0], 1, target.shape[1], target.shape[1])), -1, 0)
    combined_mask = tf.maximum(decoder_target_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask
