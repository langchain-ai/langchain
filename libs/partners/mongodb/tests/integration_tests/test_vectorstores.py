"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

import os
from time import sleep
from typing import Any, Dict, List, Optional

import pytest
from bson import ObjectId
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.utils import oid_to_str
from tests.utils import ConsistentFakeEmbeddings

INDEX_NAME = "langchain-test-index-vectorstores"
NAMESPACE = "langchain_test_db.langchain_test_vectorstores"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
DIMENSIONS = 1536
TIMEOUT = 10.0
INTERVAL = 0.5


class PatchedMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    def bulk_embed_and_insert_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = super().bulk_embed_and_insert_texts(texts, metadatas, ids)
        timeout = TIMEOUT
        while (
            len(ids_inserted) != len(self.similarity_search("sandwich"))
            and timeout >= 0
        ):
            sleep(INTERVAL)
            timeout -= INTERVAL
        return ids_inserted


def get_collection() -> Collection:
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


@pytest.fixture()
def collection() -> Collection:
    return get_collection()


@pytest.fixture
def texts() -> List[str]:
    return [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]


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

    @pytest.fixture
    def embeddings(self) -> Embeddings:
        return ConsistentFakeEmbeddings(DIMENSIONS)

    def test_from_documents(self, embeddings: Embeddings, collection: Any) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # sleep(5)  # waits for mongot to update Lucene's index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any([key.page_content == output[0].page_content for key in documents])

    def test_from_documents_no_embedding_return(
        self, embeddings: Embeddings, collection: Any
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
            embeddings,
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
        self, embeddings: Embeddings, collection: Any
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
            embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1, include_embedding=True)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all([key.metadata.get("embedding") for key in output])
        # Check for the presence of the metadata key
        assert any([key.page_content == output[0].page_content for key in documents])

    def test_from_texts(
        self, embeddings: Embeddings, collection: Collection, texts: List[str]
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # sleep(5)  # waits for mongot to update Lucene's index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1

    def test_from_texts_with_metadatas(
        self,
        embeddings: Embeddings,
        collection: Collection,
        texts: List[str],
    ) -> None:
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        metakeys = ["a", "b", "c", "d", "e"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # sleep(5)  # waits for mongot to update Lucene's index
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any([key in output[0].metadata for key in metakeys])

    def test_from_texts_with_metadatas_and_pre_filter(
        self, embeddings: Embeddings, collection: Any, texts: List[str]
    ) -> None:
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # sleep(5)  # waits for mongot to update Lucene's index
        output = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
        )
        assert output == []

    def test_mmr(self, embeddings: Embeddings, collection: Any) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # sleep(5)  # waits for mongot to update Lucene's index
        query = "foo"
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

    def test_delete(
        self, embeddings: Embeddings, collection: Any, texts: List[str]
    ) -> None:
        vectorstore = MongoDBAtlasVectorSearch(  # PatchedMongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
        clxn: Collection = vectorstore._collection
        assert clxn.count_documents({}) == 0
        ids = vectorstore.add_texts(texts)
        assert clxn.count_documents({}) == len(texts)

        deleted = vectorstore.delete(ids[-2:])
        assert deleted
        assert clxn.count_documents({}) == len(texts) - 2

        new_ids = vectorstore.add_texts(["Pigs eat stuff", "Pigs eat sandwiches"])
        assert set(new_ids).intersection(set(ids)) == set()  # new ids will be unique.
        assert isinstance(new_ids, list)
        assert all(isinstance(i, str) for i in new_ids)
        assert len(new_ids) == 2
        assert clxn.count_documents({}) == 4

    def test_add_texts(
        self,
        embeddings: Embeddings,
        collection: Collection,
        texts: List[str],
    ) -> None:
        """Tests API of add_texts, focussing on id treatment"""
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]

        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=collection, embedding=embeddings, index_name=INDEX_NAME
        )

        # Case 1. Add texts without ids
        provided_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
        all_docs = list(vectorstore._collection.find({}))
        assert all("_id" in doc for doc in all_docs)
        docids = set(doc["_id"] for doc in all_docs)
        assert all(isinstance(_id, ObjectId) for _id in docids)  #
        assert set(provided_ids) == set(oid_to_str(oid) for oid in docids)

        # Case 2: Test Document.metadata looks right. i.e. contains _id
        search_res = vectorstore.similarity_search_with_score("sandwich", k=1)
        doc, score = search_res[0]
        assert "_id" in doc.metadata

        # Case 3: Add new ids that are 24-char hex strings
        hex_ids = [oid_to_str(ObjectId()) for _ in range(2)]
        hex_texts = ["Text for hex_id"] * len(hex_ids)
        out_ids = vectorstore.add_texts(texts=hex_texts, ids=hex_ids)
        assert set(out_ids) == set(hex_ids)
        assert collection.count_documents({}) == len(texts) + len(hex_texts)
        assert all(
            isinstance(doc["_id"], ObjectId) for doc in vectorstore._collection.find({})
        )

        # Case 4: Add new ids that cannot be cast to ObjectId
        #   - We can still index and search on them
        str_ids = ["Sandwiches are beautiful,", "..sandwiches are fine."]
        str_texts = str_ids  # No reason for them to differ
        out_ids = vectorstore.add_texts(texts=str_texts, ids=str_ids)
        assert set(out_ids) == set(str_ids)
        assert collection.count_documents({}) == 8
        search_res = vectorstore.similarity_search("sandwich", k=8)
        assert any(str_ids[0] in doc.metadata["_id"] for doc in search_res)

        # Case 5: Test adding in multiple batches
        batch_size = 2
        batch_ids = [oid_to_str(ObjectId()) for _ in range(2 * batch_size)]
        batch_texts = [f"Text for batch text {i}" for i in range(2 * batch_size)]
        out_ids = vectorstore.add_texts(
            texts=batch_texts, ids=batch_ids, batch_size=batch_size
        )
        assert set(out_ids) == set(batch_ids)
        assert collection.count_documents({}) == 12

        '''
       def test_add_documents(self):
           """Tests API of add_documents, focussing on id treatment
             

           - case without ids
           - case with ids in metadata
           """
           raise NotImplementedError
           # TODO - There doesn't appear to be an add_documents!!!
        '''
