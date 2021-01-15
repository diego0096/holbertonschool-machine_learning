#!/usr/bin/env python3
""" Module used to """


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset ():
    """Loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """summary"""
        ds = 'ted_hrlr_translate/pt_to_en'
        tr = 'train'
        vl = 'validation'
        t_sample, v_sample = tfds.load(ds, with_info=True, as_supervised=True)
        self.data_train, self.data_valid = t_sample[tr], t_sample[vl]
        portuguese, english = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = portuguese, english

    def tokenize_dataset(self, data):
        """Instance method that creates sub-word tokenizers for our dataset"""
        size = (2 ** 15)
        corpus = tfds.features.text.SubwordTextEncoder.build_from_corpus
        pt = corpus((pt.numpy() for pt, en in data), target_vocab_size=size)
        en = corpus((en.numpy() for pt, en in data), target_vocab_size=size)
        return pt, en

    def encode(self, pt, en):
        """Instance method encodes a translation into token"""
        pt_a = self.tokenizer_pt.vocab_size
        pt_b = self.tokenizer_pt.encode(pt.numpy())
        pt_c = pt_a + 1
        pt_tokens = [pt_a] + pt_b + [pt_c]
        en_a = self.tokenizer_en.vocab_size
        en_b = self.tokenizer_en.encode(pt.numpy())
        en_c = en_a + 1
        en_tokens = [en_a] + en_b + [en_c]
        return pt_tokens, en_tokens
