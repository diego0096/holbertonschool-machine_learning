#!/usr/bin/env python3
"""playing with pymongo"""


def schools_by_topic(mongo_collection, topic):
    """Mongo_collection will be the pymongo collection"""
    all_items = mongo_collection.find()
    documents = []
    doc_filter = []
    for elem in all_items:
        documents.append(elem)
    for elem in documents:
        if 'topics' in elem.keys():
            if topic in elem['topics']:
                doc_filter.append(elem)
    return doc_filter