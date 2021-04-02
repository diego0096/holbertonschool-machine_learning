#!/usr/bin/env python3
""" Crop """
import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image"""
    img = tf.random_crop(image, size=size)
    return img
