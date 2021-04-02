#!/usr/bin/env python3
""" Flip """
import tensorflow as tf


def flip_image(image):
    """ flips an image horizontally"""
    flip = tf.image.flip_left_right(image)
    return flip
