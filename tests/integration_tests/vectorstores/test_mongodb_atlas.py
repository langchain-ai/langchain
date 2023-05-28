"""Test MongoDB Atlas Vector Search functionality."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

if TYPE_CHECKING:
    from pymongo import MongoClient

INDEX_NAME = "langchain-test-index"  # name of the index
NAMESPACE = "langchain_test_db.langchain_test_collection"  # name of the namespace
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")


@pytest.mark.requires("pymongo")
class TestMongoDBAtlasVectorSearch:
    @classmethod
    def test_collection(cls) -> MongoClient:
        from pymongo import MongoClient

        db_name, collection_name = NAMESPACE.split(".")
        client = MongoClient(CONNECTION_STRING)
        test_collection = client[db_name][collection_name]
        assert test_collection.count_documents({}) == 0

    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        assert cls.test_collection().count_documents({}) == 0

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the documents in the collection
        cls.test_collection().delete_many({})

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the documents in the collection
        self.test_collection().delete_many({})

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
            texts, embedding, connection_string=CONNECTION_STRING, namespace=NAMESPACE
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
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [
            Document(page_content="What is a sandwich?", metadata={"c": 1})
        ]
