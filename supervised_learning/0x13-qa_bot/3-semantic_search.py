#!/usr/bin/env python3
"""3-semantic_search.py"""


import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """3-semantic_search.py"""
    articles = [sentence]
    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as file:
            articles.append(file.read())
    embed = hub.load("https://tfhub.dev/google/universal-"
                     + "sentence-encoder-large/5")
    embeddings = embed(articles)
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    print(closest)
    return articles[closest + 1]
