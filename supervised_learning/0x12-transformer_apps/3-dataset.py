#!/usr/bin/env python3
"""Set up dataset pipeline for fitting"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
tf.compat.v1.enable_eager_execution()


class Dataset:
    """Constructor"""
    def __init__(self, batch_size, max_len):
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        pt, en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt
        self.tokenizer_en = en
        self.data_train = self.data_train.map(lambda x, y:
                                              tf.py_function(self.tf_encode,
                                                             [x, y],
                                                             (tf.int64,
                                                              tf.int64)))
        self.data_valid = self.data_valid.map(lambda x, y:
                                              tf.py_function(self.tf_encode,
                                                             [x, y],
                                                             (tf.int64,
                                                              tf.int64)))
        self.data_train = self.data_train.filter(lambda x, y: len(x) <= max_len
                                                 and len(y) <= max_len)
        self.data_valid = self.data_valid.filter(lambda x, y: len(x) <= max_len
                                                 and len(y) <= max_len)
        self.data_train = self.data_train.cache().shuffle(10000000)
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       ([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.
                                                   AUTOTUNE)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       ([None], [None]))

    def tokenize_dataset(self, data):
        """Create subword tokenizers"""
        builder = tfds.features.text.SubwordTextEncoder.build_from_corpus
        pt = builder((pt.numpy() for pt, _ in data.repeat(1)), 2**15)
        en = builder((en.numpy() for _, en in data.repeat(1)), 2**15)
        return pt, en

    def encode(self, pt, en):
        """Encode a translation pair into tokens"""
        pt = self.tokenizer_pt.encode(pt.numpy())
        en = self.tokenizer_en.encode(en.numpy())
        vocab_size = self.tokenizer_pt.vocab_size
        pt.insert(0, vocab_size)
        pt.append(vocab_size + 1)
        vocab_size = self.tokenizer_en.vocab_size
        en.insert(0, vocab_size)
        en.append(vocab_size + 1)
        return pt, en

    def tf_encode(self, pt, en):
        """Tensorflow dataset wrapper for encoding"""
        pt, en = self.encode(pt, en)
        pt = tf.cast(pt, tf.int64)
        en = tf.cast(en, tf.int64)
        return tf.convert_to_tensor(pt), tf.convert_to_tensor(en)
