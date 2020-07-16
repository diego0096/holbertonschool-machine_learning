#!/usr/bin/env python3
"""add two arrays elements"""


def add_arrays(arr1, arr2):
    """add elements"""
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
