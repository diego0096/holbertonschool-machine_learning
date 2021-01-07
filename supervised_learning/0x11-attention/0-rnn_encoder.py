#!/usr/bin/env python3
"""0-rnn_encoder.py"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class that instantiates a RNN Encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """function that initializes the hidden state to a tensor of zeros"""
        initializer = tf.keras.initializers.Zeros()
        tensor = initializer(shape=(self.batch, self.units))
        return tensor

    def call(self, x, initial):
        """function that builds the encoder"""
        embeddings = self.embedding(x)
        full_seq_outputs, last_hidden_state = self.gru(embeddings,
                                                       initial_state=initial)
        return full_seq_outputs, last_hidden_state
