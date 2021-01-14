#!/usr/bin/env python3
"""0-dataset.py"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """class that loads and preps a dataset for machine translation"""

    def __init__(self):
        """constructor"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """function that creates sub-word tokenizers for a dataset"""
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en