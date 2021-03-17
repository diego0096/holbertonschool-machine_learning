#!/usr/bin/env python3
"""insert a document in a collection"""


def insert_school(mongo_collection, **kwargs):
    """insert in mongo collection a new documnet"""
    mongo_collection.insert(kwargs)
    new_elem = mongo_collection.find(kwargs)
    return new_elem.__dict__['_Cursor__spec']['_id']
