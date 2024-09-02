"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

import os
from time import monotonic, sleep
from typing import Any, Dict, List

import pytest  # type: ignore[import-not-found]
from bson import ObjectId
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from langchain_mongodb.index import drop_vector_search_index
from langchain_mongodb.utils import oid_to_str

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

INDEX_NAME = "langchain-test-index-vectorstores"
INDEX_CREATION_NAME = "langchain-test-index-vectorstores-create-test"
NAMESPACE = "langchain_test_db.langchain_test_vectorstores"
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
INDEX_COLLECTION_NAME = "langchain_test_vectorstores_index"
INDEX_DB_NAME = "langchain_test_index_db"
DIMENSIONS = 1536
TIMEOUT = 120.0
INTERVAL = 0.5


@pytest.fixture
def example_documents() -> List[Document]:
    return [
        Document(page_content="Dogs are tough.", metadata={"a": 1}),
        Document(page_content="Cats have fluff.", metadata={"b": 1}),
        Document(page_content="What is a sandwich?", metadata={"c": 1}),
        Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
    ]


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


@pytest.fixture
def texts() -> List[str]:
    return [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]


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
    def embeddings(self) -> Embeddings:
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
                model="text-embedding-3-small",
            )
        except Exception:
            return ConsistentFakeEmbeddings(DIMENSIONS)

    def test_from_documents(
        self,
        embeddings: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_documents_no_embedding_return(
        self,
        embeddings: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all(["embedding" not in key.metadata for key in output])
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_documents_embedding_return(
        self,
        embeddings: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test end to end construction and search."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        output = vectorstore.similarity_search("Sandwich", k=1, include_embeddings=True)
        assert len(output) == 1
        # Check for presence of embedding in each document
        assert all([key.metadata.get("embedding") for key in output])
        # Check for the presence of the metadata key
        assert any(
            [key.page_content == output[0].page_content for key in example_documents]
        )

    def test_from_texts(
        self, embeddings: Embeddings, collection: Collection, texts: List[str]
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
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
            embedding=embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
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
            embedding=embeddings,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        does_not_match_filter = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
        )
        assert does_not_match_filter == []

        matches_filter = vectorstore.similarity_search(
            "Sandwich", k=3, pre_filter={"c": {"$gt": 0}}
        )
        assert len(matches_filter) == 1

    def test_mmr(self, embeddings: Embeddings, collection: Any) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        query = "foo"
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

    def test_retriever(
        self,
        embeddings: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Demonstrate usage and parity of VectorStore similarity_search
        with Retriever.invoke."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            example_documents,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )
        query = "sandwich"

        retriever_default_kwargs = vectorstore.as_retriever()
        result_retriever = retriever_default_kwargs.invoke(query)
        result_vectorstore = vectorstore.similarity_search(query)
        assert all(
            [
                result_retriever[i].page_content == result_vectorstore[i].page_content
                for i in range(len(result_retriever))
            ]
        )

    def test_include_embeddings(
        self,
        embeddings: Embeddings,
        collection: Any,
        example_documents: List[Document],
    ) -> None:
        """Test explicitly passing vector kwarg matches default."""
        vectorstore = PatchedMongoDBAtlasVectorSearch.from_documents(
            documents=example_documents,
            embedding=embeddings,
            collection=collection,
            index_name=INDEX_NAME,
        )

        output_with = vectorstore.similarity_search(
            "Sandwich", include_embeddings=True, k=1
        )
        assert vectorstore._embedding_key in output_with[0].metadata
        output_without = vectorstore.similarity_search("Sandwich", k=1)
        assert vectorstore._embedding_key not in output_without[0].metadata

    def test_delete(
        self, embeddings: Embeddings, collection: Any, texts: List[str]
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
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
        """Tests API of add_texts, focussing on id treatment

        Warning: This is slow because of the number of cases
        """
        metadatas: List[Dict[str, Any]] = [
            {"a": 1},
            {"b": 1},
            {"c": 1},
            {"d": 1, "e": 2},
        ]

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
        res = vectorstore.similarity_search("sandwich", k=8)
        assert any(str_ids[0] in doc.metadata["_id"] for doc in res)

        # Case 5: Test adding in multiple batches
        batch_size = 2
        batch_ids = [oid_to_str(ObjectId()) for _ in range(2 * batch_size)]
        batch_texts = [f"Text for batch text {i}" for i in range(2 * batch_size)]
        out_ids = vectorstore.add_texts(
            texts=batch_texts, ids=batch_ids, batch_size=batch_size
        )
        assert set(out_ids) == set(batch_ids)
        assert collection.count_documents({}) == 12

        # Case 6: _ids in metadata
        collection.delete_many({})
        # 6a. Unique _id in metadata, but ids=None
        # Will be added as if ids kwarg provided
        i = 0
        n = len(texts)
        assert len(metadatas) == n
        _ids = [str(i) for i in range(n)]
        for md in metadatas:
            md["_id"] = _ids[i]
            i += 1
        returned_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
        assert returned_ids == ["0", "1", "2", "3"]
        assert set(d["_id"] for d in vectorstore._collection.find({})) == set(_ids)

        # 6b. Unique "id", not "_id", but ids=None
        # New ids will be assigned
        i = 1
        for md in metadatas:
            md.pop("_id")
            md["id"] = f"{1}"
            i += 1
        returned_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
        assert len(set(returned_ids).intersection(set(_ids))) == 0

    def test_add_documents(
        self,
        embeddings: Embeddings,
        collection: Collection,
    ) -> None:
        """Tests add_documents."""
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=collection, embedding=embeddings, index_name=INDEX_NAME
        )

        # Case 1: No ids
        n_docs = 10
        batch_size = 3
        docs = [
            Document(page_content=f"document {i}", metadata={"i": i})
            for i in range(n_docs)
        ]
        result_ids = vectorstore.add_documents(docs, batch_size=batch_size)
        assert len(result_ids) == n_docs
        assert collection.count_documents({}) == n_docs

        # Case 2: ids
        collection.delete_many({})
        n_docs = 10
        batch_size = 3
        docs = [
            Document(page_content=f"document {i}", metadata={"i": i})
            for i in range(n_docs)
        ]
        ids = [str(i) for i in range(n_docs)]
        result_ids = vectorstore.add_documents(docs, ids, batch_size=batch_size)
        assert len(result_ids) == n_docs
        assert set(ids) == set(collection.distinct("_id"))

        # Case 3: Single batch
        collection.delete_many({})
        n_docs = 3
        batch_size = 10
        docs = [
            Document(page_content=f"document {i}", metadata={"i": i})
            for i in range(n_docs)
        ]
        ids = [str(i) for i in range(n_docs)]
        result_ids = vectorstore.add_documents(docs, ids, batch_size=batch_size)
        assert len(result_ids) == n_docs
        assert set(ids) == set(collection.distinct("_id"))

    def test_index_creation(
        self, embeddings: Embeddings, index_collection: Any
    ) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            index_collection, embedding=embeddings, index_name=INDEX_CREATION_NAME
        )
        vectorstore.create_vector_search_index(dimensions=1536)

    def test_index_update(self, embeddings: Embeddings, index_collection: Any) -> None:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            index_collection, embedding=embeddings, index_name=INDEX_CREATION_NAME
        )
        vectorstore.create_vector_search_index(dimensions=1536)
        vectorstore.create_vector_search_index(dimensions=1536, update=True)
