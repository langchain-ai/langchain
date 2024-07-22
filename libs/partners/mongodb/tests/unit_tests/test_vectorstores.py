from json import dumps, loads
from typing import Any, Optional

import pytest  # type: ignore[import-not-found]
from bson import ObjectId, json_util
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.utils import str_to_oid
from tests.utils import ConsistentFakeEmbeddings, MockCollection

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def get_collection() -> MockCollection:
    return MockCollection()


@pytest.fixture()
def collection() -> MockCollection:
    return get_collection()


@pytest.fixture(scope="module")
def embedding_openai() -> Embeddings:
    return ConsistentFakeEmbeddings()


def test_initialization(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test initialization of vector store class"""
    assert MongoDBAtlasVectorSearch(collection, embedding_openai)


def test_init_from_texts(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test from_texts operation on an empty list"""
    assert MongoDBAtlasVectorSearch.from_texts(
        [], embedding_openai, collection=collection
    )


class TestMongoDBAtlasVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # ensure the test collection is empty
        collection = get_collection()
        assert collection.count_documents({}) == 0  # type: ignore[index]

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

    def _validate_search(
        self,
        vectorstore: MongoDBAtlasVectorSearch,
        collection: MockCollection,
        search_term: str = "sandwich",
        page_content: str = "What is a sandwich?",
        metadata: Optional[Any] = 1,
    ) -> None:
        collection._aggregate_result = list(
            filter(
                lambda x: search_term.lower() in x[vectorstore._text_key].lower(),
                collection._data,
            )
        )
        output = vectorstore.similarity_search("", k=1)
        assert output[0].page_content == page_content
        assert output[0].metadata.get("c") == metadata
        # Validate the ObjectId provided is json serializable
        assert loads(dumps(output[0].page_content)) == output[0].page_content
        assert loads(dumps(output[0].metadata)) == output[0].metadata
        json_metadata = dumps(output[0].metadata)  # normal json.dumps
        isinstance(str_to_oid(json_util.loads(json_metadata)["_id"]), ObjectId)

    def test_from_documents(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        self._validate_search(
            vectorstore, collection, metadata=documents[2].metadata["c"]
        )

    def test_from_texts(
        self, embedding_openai: Embeddings, collection: MockCollection
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
            collection=collection,
            index_name=INDEX_NAME,
        )
        self._validate_search(vectorstore, collection, metadata=None)

    def test_from_texts_with_metadatas(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        self._validate_search(vectorstore, collection, metadata=metadatas[2]["c"])

    def test_from_texts_with_metadatas_and_pre_filter(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        collection._aggregate_result = list(
            filter(
                lambda x: "sandwich" in x[vectorstore._text_key].lower()
                and x.get("c") < 0,
                collection._data,
            )
        )
        output = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"range": {"lte": 0, "path": "c"}}
        )
        assert output == []

    def test_mmr(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        query = "foo"
        self._validate_search(
            vectorstore,
            collection,
            search_term=query[0:2],
            page_content=query,
            metadata=None,
        )
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"
