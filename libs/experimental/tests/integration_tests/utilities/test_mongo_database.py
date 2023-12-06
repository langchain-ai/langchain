"""Test MongoDB database wrapper."""

import re

from pymongo import MongoClient

from langchain_experimental.utilities.mongo_database import MongoDatabase

uri = "mongodb://%2Ftmp%2Fmongodb-27017.sock/test_db?inMemory=true"


def test_collection_info() -> None:
    """Test that collection info is constructed properly."""
    db = MongoDatabase.from_uri(uri)
    collection = db._client["test_db"]["test_collection"]

    if "test" not in collection.find_one({"test": "test"}):  # type: ignore
        collection.insert_many(
            [
                {"test": "test"},
                {"test2": "test"},
                {"test3": "test"},
                {"test4": "test"},
            ]
        )
    output = db.collection_info
    expected_output = """
    Collection Name: test_collection

    3 sample documents from test_collection:
    {'_id': , 'test': 'test'}
    {'_id': , 'test2': 'test'}
    {'_id': , 'test3': 'test'}
    """
    output = re.sub(r"ObjectId\('.+'\)", "", output)

    assert sorted(" ".join(output.split())) == sorted(" ".join(expected_output.split()))


def test_collection_info_w_sample_documents() -> None:
    """Test that collection info is constructed properly."""
    db = MongoDatabase(
        MongoClient(uri),
        sample_documents_in_collection_info=2,
    )
    collection = db._client["test_db"]["test_collection"]

    if "test" not in collection.find_one({"test": "test"}):  # type: ignore
        collection.insert_many(
            [
                {"test": "test"},
                {"test2": "test"},
                {"test3": "test"},
                {"test4": "test"},
            ]
        )
    output = db.collection_info
    expected_output = """
    Collection Name: test_collection

    2 sample documents from test_collection:
    {'_id': , 'test': 'test'}
    {'_id': , 'test2': 'test'}
    """
    output = re.sub(r"ObjectId\('.+'\)", "", output)

    assert sorted(" ".join(output.split())) == sorted(" ".join(expected_output.split()))


def test_mongo_database_run() -> None:
    """Test that run works properly."""
    db = MongoDatabase.from_uri(uri)
    output = db.run("{ 'find': 'test_collection', 'filter': { 'test4': 'test' } }")
    expected_output = """
    Result:
    {'_id': , 'test4': 'test'}
    """
    output = re.sub(r"ObjectId\('.+'\)", "", output)

    assert sorted(" ".join(output.split())) == sorted(" ".join(expected_output.split()))
