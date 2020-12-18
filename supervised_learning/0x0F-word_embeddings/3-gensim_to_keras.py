#!/usr/bin/env python3
"""3-gensim_to_keras.py"""


from gensim.models import Word2Vec


def gensim_to_keras(model):
    """function that converts a gensim word2vec"""
    return model.wv.get_keras_embedding(train_embeddings=True)
