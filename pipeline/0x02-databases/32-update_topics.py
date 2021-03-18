#!/usr/bin/env python3
"""update collection"""


def update_topics(mongo_collection, name, topics):
    """update documents of collection"""
    aux = {'$set': {'topics': topics}}
    mongo_collection.update_many({'name': name}, aux)
