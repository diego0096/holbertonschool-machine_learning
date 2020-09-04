#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a valid convolution on grayscale images"""
    n_images = images.shape[0]
    i_h = images.shape[1]
    i_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    s_h = stride[0]
    s_w = stride[1]
    p_h = 0
    p_w = 0
    if (padding == "same"):
        p_h = int(((i_h - 1) * s_h + k_h - i_h) / 2) + 1
        p_w = int(((i_w - 1) * s_w + k_w - i_w) / 2) + 1
    elif (isinstance(padding, tuple)):
        p_h = padding[0]
        p_w = padding[1]
    o_h = np.floor(((i_h + 2 * p_h - k_h) / s_h) + 1).astype(int)
    o_w = np.floor(((i_w + 2 * p_w - k_w) / s_w) + 1).astype(int)
    outputs = np.zeros((n_images, o_h, o_w))
    padded_imgs = np.pad(images,
                         ((0, 0), (p_h, p_h), (p_w, p_w)),
                         mode="constant",
                         constant_values=0)
    imgs_arr = np.arange(0, n_images)
    for x in range(o_h):
        for y in range(o_w):
            x0 = x * s_h
            y0 = y * s_w
            x1 = x0 + k_h
            y1 = y0 + k_w
            outputs[imgs_arr, x, y] = np.sum(np.multiply(
                padded_imgs[imgs_arr, x0: x1, y0: y1], kernel), axis=(1, 2))
    return outputs
