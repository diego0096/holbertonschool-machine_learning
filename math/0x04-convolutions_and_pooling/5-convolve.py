#!/usr/bin/env python3
"""Module used to"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a valid convolution on images with multiple channels"""
    n_images = images.shape[0]
    i_h = images.shape[1]
    i_w = images.shape[2]
    i_c = images.shape[3]
    k_h = kernels.shape[0]
    k_w = kernels.shape[1]
    k_c = kernels.shape[3]
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
    o_h = int((i_h + 2 * p_h - k_h) / s_h) + 1
    o_w = int((i_w + 2 * p_w - k_w) / s_w) + 1
    outputs = np.zeros((n_images, o_h, o_w, k_c))
    padded_imgs = np.pad(images,
                         ((0, 0),
                          (p_h, p_h),
                          (p_w, p_w),
                          (0, 0)
                          ),
                         mode="constant",
                         constant_values=0)
    for x in range(o_h):
        for y in range(o_w):
            for z in range(k_c):
                x0 = x * s_h
                y0 = y * s_w
                x1 = x0 + k_h
                y1 = y0 + k_w
                outputs[imgs_arr, x, y, z] = np.sum(np.multiply(
                    padded_imgs[imgs_arr, x0: x1, y0: y1],
                    kernels[:, :, :, z]), axis=(1, 2, 3))
    return outputs
