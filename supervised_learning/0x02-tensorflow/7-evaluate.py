#!/usr/bin/env python3
"""Evalute restored model"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate restored model"""
    sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("layer_2/BiasAdd:0")
    accuracy = graph.get_tensor_by_name("Mean:0")
    loss = graph.get_tensor_by_name("softmax_cross_entropy_loss/value:0")
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    return sess.run((y_pred, accuracy, loss), feed_dict={x: X, y: Y})
