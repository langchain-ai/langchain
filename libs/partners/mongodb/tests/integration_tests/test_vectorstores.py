"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

import os
from time import monotonic, sleep
from typing import Any, Dict, List, Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import drop_vector_search_index
from tests.utils import ConsistentFakeEmbeddings

INDEX_NAME = "langchain-test-index-vectorstores"
INDEX_CREATION_NAME = "langchain-test-index-vectorstores-create-test"
NAMESPACE = "langchain_test_db.langchain_test_vectorstores"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
INDEX_COLLECTION_NAME = "langchain_test_vectorstores_index"
INDEX_DB_NAME = "langchain_test_index_db"
DIMENSIONS = 1536
TIMEOUT = 10.0
INTERVAL = 0.5


class PatchedMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    def _insert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids = super()._insert_texts(texts, metadatas)
        start = monotonic()
        while len(ids) != self.similarity_search("sandwich") and (
            monotonic() - start <= TIMEOUT
        ):
            sleep(INTERVAL)
        return ids

    def create_vector_search_index(
        self,
        dimensions: int,
        filters: Optional[List[Dict[str, str]]] = None,
        update: bool = False,
    ) -> None:
        result = super().create_vector_search_index(
            dimensions=dimensions, filters=filters, update=update
        )
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if indexes := list(
                self._collection.list_search_indexes(name=self._index_name)
            ):
                if indexes[0].get("status") == "READY":
                    return result
            sleep(INTERVAL)

        raise TimeoutError(f"{self._index_name} never reached 'status: READY'")


def _await_index_deletion(coll: Collection, index_name: str) -> None:
    start = monotonic()
    try:
        drop_vector_search_index(coll, index_name)
    except OperationFailure:
        # This most likely means an ongoing drop request was made so skip
        pass

    while list(coll.list_search_indexes(name=index_name)):
        if monotonic() - start > TIMEOUT:
            raise TimeoutError(f"Index Name: {index_name} never dropped")
        sleep(INTERVAL)


def get_collection(
    database_name: str = DB_NAME, collection_name: str = COLLECTION_NAME
) -> Collection:
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[database_name][collection_name]


@pytest.fixture()
def collection() -> Collection:
    return get_collection()


@pytest.fixture()
def index_collection() -> Collection:
    return get_collection(INDEX_DB_NAME, INDEX_COLLECTION_NAME)


class TestMongoDBAtlasVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # insure the test collection is empty
        collection = get_collection()
        if collection.count_documents({}):
            collection.delete_many({})  # type: ignore[index]

    @classmethod
    def teardown_class(cls) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

        # delete all indexes on index collection name
        _await_index_deletion(
            get_collection(INDEX_DB_NAME, INDEX_COLLECTION_NAME), INDEX_CREATION_NAME
        )

    @pytest.fixture
    def embedding_openai(self) -> Embeddings:
        return ConsistentFakeEmbeddings(DIMENSIONS)

    def test_from_documents(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any([key.page_content == output[0].page_content for key in documents])

    def test_from_documents_no_embedding_return(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all(["embedding" not in key.metadata for key in output])
        # Check for the presence of the metadata key
        assert any([key.page_content == output[0].page_content for key in documents])

    def test_from_documents_embedding_return(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1, include_embedding=True)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all([key.metadata.get("embedding") for key in output])
        # Check for the presence of the metadata key
        assert any([key.page_content == output[0].page_content for key in documents])

    def test_from_texts(self, embedding_openai: Embeddings, collection: Any) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1

    def test_from_texts_with_metadatas(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        metakeys = ["a", "b", "c", "d", "e"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any([key in output[0].metadata for key in metakeys])

    def test_from_texts_with_metadatas_and_pre_filter(
        self, embedding_openai: Embeddings, collection: Any
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
        )
        assert output == []

    def test_mmr(self, embedding_openai: Embeddings, collection: Any) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        query = "foo"
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

    def test_index_creation(
        self, embedding_openai: Embeddings, index_collection: Any
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            index_collection, embedding_openai, index_name=INDEX_CREATION_NAME
        )
        vectorstore.create_vector_search_index(dimensions=1536)

    def test_index_update(
        self, embedding_openai: Embeddings, index_collection: Any
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            index_collection, embedding_openai, index_name=INDEX_CREATION_NAME
        )
        vectorstore.create_vector_search_index(dimensions=1536)
        vectorstore.create_vector_search_index(dimensions=1536, update=True)
