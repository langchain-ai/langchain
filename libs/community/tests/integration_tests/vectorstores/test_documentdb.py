"""Test DocumentDBVectorSearch functionality."""

import logging
import os
from time import sleep
from typing import Any, Optional

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.documentdb import (
    DocumentDBSimilarityType,
    DocumentDBVectorSearch,
)

logging.basicConfig(level=logging.DEBUG)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING = os.getenv("DOCUMENTDB_URI", "")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

dimensions = 1536
similarity_algorithm = DocumentDBSimilarityType.COS


def prepare_collection() -> Any:
    from pymongo import MongoClient

    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Any:
    return prepare_collection()


@pytest.fixture()
def embedding_openai() -> Any:
    openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        deployment=model_deployment, model=model_name, chunk_size=1
    )
    return openai_embeddings


"""
This is how to run the integration tests:

cd libs/community
make test TEST_FILE=tests/integration_tests/vectorstores/test_documentdb.py

NOTE: You will first need to follow the contributor setup steps:
https://python.langchain.com/docs/contributing/code. You will also need to install
`pymongo` via `poetry`. You can also run the test directly using `pytest`, but please
make sure to install all dependencies.
"""


class TestDocumentDBVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # insure the test collection is empty
        collection = prepare_collection()
        assert collection.count_documents({}) == 0  # type: ignore[index]

    @classmethod
    def teardown_class(cls) -> None:
        collection = prepare_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]
        collection.drop_indexes()

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = prepare_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]
        collection.drop_indexes()

    def test_from_documents_cosine_distance(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = DocumentDBVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        sleep(1)  # waits for DocumentDB to save contents to the collection

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, similarity_algorithm)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        vectorstore.delete_index()

    def test_from_documents_inner_product(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]

        vectorstore = DocumentDBVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        sleep(1)  # waits for DocumentDB to save contents to the collection

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, DocumentDBSimilarityType.DOT)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1, ef_search=100)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        vectorstore.delete_index()

    def test_from_texts_cosine_distance(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, similarity_algorithm)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output[0].page_content == "What is a sandwich?"
        vectorstore.delete_index()

    def test_from_texts_with_metadatas_cosine_distance(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, similarity_algorithm)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_one(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, similarity_algorithm)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1

        first_document_id_object = output[0].metadata["_id"]
        first_document_id = str(first_document_id_object)

        vectorstore.delete_document_by_id(first_document_id)
        sleep(2)  # waits for the index to be updated

        output2 = vectorstore.similarity_search("Sandwich", k=1, ef_search=10)
        assert output2
        assert output2[0].page_content != "What is a sandwich?"

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_delete_multiple(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, similarity_algorithm)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=5)

        first_document_id_object = output[0].metadata["_id"]
        first_document_id = str(first_document_id_object)

        output[1].metadata["_id"]
        second_document_id = output[1].metadata["_id"]

        output[2].metadata["_id"]
        third_document_id = output[2].metadata["_id"]

        document_ids = [first_document_id, second_document_id, third_document_id]
        vectorstore.delete(document_ids)
        sleep(2)  # waits for the index to be updated

        output_2 = vectorstore.similarity_search("Sandwich", k=5)
        assert output
        assert output_2

        assert len(output) == 4  # we should see all the four documents
        assert (
            len(output_2) == 1
        )  # we should see only one document left after three have been deleted

        vectorstore.delete_index()

    def test_from_texts_with_metadatas_inner_product(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, DocumentDBSimilarityType.DOT)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        vectorstore.delete_index()

    def test_from_texts_with_metadatas_euclidean_distance(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = DocumentDBVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )

        # Create the HNSW index that will be leveraged later for vector search
        vectorstore.create_index(dimensions, DocumentDBSimilarityType.EUC)
        sleep(2)  # waits for the index to be set up

        output = vectorstore.similarity_search("Sandwich", k=1)

        assert output
        assert output[0].page_content == "What is a sandwich?"
        assert output[0].metadata["c"] == 1
        vectorstore.delete_index()

    def invoke_delete_with_no_args(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> Optional[bool]:
        vectorstore: DocumentDBVectorSearch = (
            DocumentDBVectorSearch.from_connection_string(
                CONNECTION_STRING,
                NAMESPACE,
                embedding_openai,
                index_name=INDEX_NAME,
            )
        )

        return vectorstore.delete()

    def invoke_delete_by_id_with_no_args(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        vectorstore: DocumentDBVectorSearch = (
            DocumentDBVectorSearch.from_connection_string(
                CONNECTION_STRING,
                NAMESPACE,
                embedding_openai,
                index_name=INDEX_NAME,
            )
        )

        vectorstore.delete_document_by_id()

    def test_invalid_arguments_to_delete(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(ValueError) as exception_info:
            self.invoke_delete_with_no_args(embedding_openai, collection)
        assert str(exception_info.value) == "No document ids provided to delete."

    def test_no_arguments_to_delete_by_id(
        self, embedding_openai: OpenAIEmbeddings, collection: Any
    ) -> None:
        with pytest.raises(Exception) as exception_info:
            self.invoke_delete_by_id_with_no_args(embedding_openai, collection)
        assert str(exception_info.value) == "No document id provided to delete."
