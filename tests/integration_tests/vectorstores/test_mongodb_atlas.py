"""Test MongoDB Atlas Vector Search functionality."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import pytest

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

if TYPE_CHECKING:
    from pymongo import MongoClient

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def get_test_client() -> Optional[MongoClient]:
    try:
        from pymongo import MongoClient

        client = MongoClient(CONNECTION_STRING)
        return client
    except:  # noqa: E722
        return None


# Instantiate as constant instead of pytest fixture to prevent needing to make multiple
# connections.
TEST_CLIENT = get_test_client()


class TestMongoDBAtlasVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        assert TEST_CLIENT[DB_NAME][COLLECTION_NAME].count_documents({}) == 0  # type: ignore[index]  # noqa: E501

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the documents in the collection
        TEST_CLIENT[DB_NAME][COLLECTION_NAME].delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the documents in the collection
        TEST_CLIENT[DB_NAME][COLLECTION_NAME].delete_many({})  # type: ignore[index]

    @pytest.mark.vcr()
    def test_from_documents(self, embedding: Embeddings) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding,
            connection_string=CONNECTION_STRING,
            namespace=NAMESPACE,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [
            Document(page_content="What is a sandwich?", metadata={"c": 1})
        ]

    @pytest.mark.vcr()
    def test_from_texts(self, embedding: Embeddings) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding,
            connection_string=CONNECTION_STRING,
            namespace=NAMESPACE,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [Document(page_content="What is a sandwich?")]

    @pytest.mark.vcr()
    def test_from_texts_with_metadatas(self, embedding: Embeddings) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            connection_string=CONNECTION_STRING,
            namespace=NAMESPACE,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [
            Document(page_content="What is a sandwich?", metadata={"c": 1})
        ]
