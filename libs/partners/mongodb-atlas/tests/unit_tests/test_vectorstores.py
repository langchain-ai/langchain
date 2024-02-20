import os

import pytest
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb_atlas.vectorstores import MongoDBAtlasVectorSearch

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def get_collection() -> Collection:
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Collection:
    return get_collection()


def test_initialization(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test initialization of vector store class"""
    assert MongoDBAtlasVectorSearch(collection, embedding_openai)


def test_init_from_connection_string(embedding_openai: Embeddings) -> None:
    """Test initialization of vector store class"""
    assert MongoDBAtlasVectorSearch.from_connection_string(
        CONNECTION_STRING, NAMESPACE, embedding_openai
    )


def test_init_from_texts(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test from_texts operation on an empty list"""
    assert MongoDBAtlasVectorSearch.from_texts(
        [], embedding_openai, collection=collection
    )
