#!/usr/bin/env python3
"""Hue"""
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image"""
    img = tf.image.adjust_hue(image, delta)
    return img
