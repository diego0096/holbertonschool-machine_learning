#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a valid convolution on grayscale images"""
    n_images = images.shape[0]
    i_h = images.shape[1]
    i_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    p_h = int((k_h - 1) / 2)
    p_w = int((k_w - 1) / 2)
    if k_h % 2 == 0:
        p_h = int(k_h / 2)
    if k_w % 2 == 0:
        p_w = int(k_w / 2)
    o_h = i_h + 2 * p_h - k_h + 1
    o_w = i_w + 2 * p_w - k_w + 1
    if k_h % 2 == 0:
        o_h = i_h + 2 * p_h - k_h
    if k_w % 2 == 0:
        o_w = i_w + 2 * p_w - k_w
    outputs = np.zeros((n_images, o_h, o_w))
    padded_imgs = np.pad(images,
                         pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
                         mode="constant",
                         constant_values=0)
    imgs_arr = np.arange(0, n_images)
    for x in range(o_h):
        for y in range(o_w):
            x1 = x + k_h
            y1 = y + k_w
            outputs[imgs_arr, x, y] = np.sum(np.multiply(
                padded_imgs[imgs_arr, x: x1, y: y1], kernel), axis=(1, 2))
    return outputs
