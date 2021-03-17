#!/usr/bin/env python3
"""Getting all documents from mongodb"""


def list_all(mongo_collection):
    """Mongo_collection: input collection"""
    documents = []
    list_all = mongo_collection.find()
    for elem in list_all:
        documents.append(elem)
    return documents
