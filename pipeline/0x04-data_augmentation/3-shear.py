#!/usr/bin/env python3
"""rotate"""


import tensorflow as tf


def shear_image(image, intensity):
    """Randomly shears an image"""
    image = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(image,
                                                        intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    return tf.keras.preprocessing.image.array_to_img(sheared)
