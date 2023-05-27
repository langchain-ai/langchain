"""Test MongoDB Atlas Vector Search functionality."""

import os
from typing import List, Dict

from pymongo import MongoClient
import pytest

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas_vector_search import MongoDBAtlasVectorSearch

index_name = "langchain-test-index"  # name of the index
namespace = "langchain_test_db.langchain_test_collection"  # name of the namespace
dimension = 1536  # dimension of the embeddings
connection_string = os.environ.get("MONGODB_ATLAS_URI")
assert connection_string is not None
client = MongoClient(connection_string)
db_name, collection_name = namespace.split('.')
test_collection = client[db_name][collection_name]


class TestMongoDBAtlasVectorSearch:

    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        assert test_collection.count_documents({}) == 0

    @classmethod
    def teardown_class(cls) -> None:
        # delete all the documents in the collection
        test_collection.delete_many({})

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the documents in the collection
        test_collection.delete_many({})

    @pytest.mark.vcr()
    def test_from_documents(
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(pageContent="Dogs are tough.", metadata={"a": 1}),
            Document(pageContent="Cats have fluff.", metadata={"b": 1}),
            Document(pageContent="What is a sandwich?", metadata={"c": 1}),
            Document(pageContent="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            client=client,
            namespace=namespace
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [Document(pageContent="What is a sandwich?", metadata={"c": 1})]

    @pytest.mark.vcr()
    def test_from_texts(
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            client=client,
            namespace=namespace
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [Document(pageContent="What is a sandwich?")]

    @pytest.mark.vcr()
    def test_from_texts_with_metadatas(
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        texts_with_metadatas = [
            {"pageContent": "Dogs are tough.", "metadata": {"a": 1}},
            {"pageContent": "Cats have fluff.", "metadata": {"b": 1}},
            {"pageContent": "What is a sandwich?", "metadata": {"c": 1}},
            {"pageContent": "That fence is purple.", "metadata": {"d": 1, "e": 2}},
        ]
        texts: List[str] = [d["pageContent"] for d in texts_with_metadatas]
        metadatas: List[Dict[str, int]] = [d["metadata"] for d in texts_with_metadatas]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas,
            client=client,
            namespace=namespace
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert output == [Document(pageContent="What is a sandwich?", metadata={"c": 1})]
