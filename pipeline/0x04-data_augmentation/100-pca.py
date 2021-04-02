#!/usr/bin/env python3
"""PCA COLOR"""
import tensorflow as tf
import numpy as np


def pca_color(img, alpha):
    """Performs PCA color augmentation"""
    img = tf.keras.preprocessing.image.img_to_array(img)
    orig_img = img.astype(float).copy()

    img = img / 255.0

    img_rs = img.reshape(-1, 3)

    img_centered = img_rs - np.mean(img_rs, axis=0)

    img_cov = np.cov(img_centered, rowvar=False)

    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    m1 = np.column_stack((eig_vecs))

    m2 = np.zeros((3, 1))

    m2[:, 0] = alpha * eig_vals[:]

    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)

    orig_img = orig_img.astype(np.uint8)

    return orig_img
