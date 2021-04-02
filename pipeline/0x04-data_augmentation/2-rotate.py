#!/usr/bin/env python3
""" Rotate """
import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees"""
    img_90 = tf.image.rot90(image, k=1)
    return img_90
