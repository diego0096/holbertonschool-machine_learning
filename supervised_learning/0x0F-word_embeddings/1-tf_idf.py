#!/usr/bin/env python3
"""1-tf_idf.py"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """function that produces an array of embeddings from a corpus"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
