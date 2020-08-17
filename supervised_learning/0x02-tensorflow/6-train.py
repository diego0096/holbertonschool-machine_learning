#!/usr/bin/env python3
"""
Train our network
"""


import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path='/tmp/model.ckpt'):
    """
    Train our network
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train = create_train_op(loss, alpha)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for iters in range(0, iterations):
        if not iters % 100:
            print('After {} iterations:'.format(iters))
            trainloss, trainaccuracy = sess.run((loss, accuracy),
                                                feed_dict={x: X_train,
                                                           y: Y_train})
