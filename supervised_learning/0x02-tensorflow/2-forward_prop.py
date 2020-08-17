#!/usr/bin/env python3
"""Create basic forward"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create basic forward"""
    first_layer = create_layer(x, layer_sizes[0], activations[0])
    last_layer = first_layer
    for layer in range(1, len(layer_sizes)):
        last_layer = create_layer(last_layer, layer_sizes[layer],
                                  activations[layer])
    return last_layer
