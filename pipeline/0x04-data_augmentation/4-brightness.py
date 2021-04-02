#!/usr/bin/env python3
"""Brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image"""
    img = tf.image.adjust_brightness(image, max_delta)
    return img
