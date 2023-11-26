"""Test MongoDB database wrapper."""

# import pytest
# from pymongo import MongoClient
import re

from langchain.utilities.mongo_database import MongoDBDatabase

db = MongoDBDatabase.from_uri("mongodb://localhost/test_db")
collection = db._client["test_db"]["test_collection"]

if "test" not in collection.find_one({"test": "test"}):
    collection.insert_many(
        [
            {"test": "test"},
            {"test2": "test"},
            {"test3": "test"},
            {"test4": "test"},
        ]
    )


def test_collection_info() -> None:
    """Test that collection info is constructed properly."""
    output = db.collection_info
    expected_output = r"""
    Collection Name: test_collection

    3 sample documents from test_collection:
    {'_id': ObjectId('.+'), 'test': 'test'}
    {'_id': ObjectId('.+'), 'test2': 'test'}
    {'_id': ObjectId('.+'), 'test3': 'test'}
    """
    assert re.match(expected_output, output)


# test_collection_info()
