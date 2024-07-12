from typing import Generator

import pytest
from langchain_core.documents import Document

from langchain_community.storage.mongodb import MongoDBByteStore, MongoDBStore

pytest.importorskip("pymongo")


@pytest.fixture
def mongo_store() -> Generator:
    import mongomock

    # mongomock creates a mock MongoDB instance for testing purposes
    with mongomock.patch(servers=(("localhost", 27017),)):
        yield MongoDBStore("mongodb://localhost:27017/", "test_db", "test_collection")


def test_mset_and_mget(mongo_store: MongoDBStore) -> None:
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")

    # Set documents in the store
    mongo_store.mset([("key1", doc1), ("key2", doc2)])

    # Get documents from the store
    retrieved_docs = mongo_store.mget(["key1", "key2"])

    assert retrieved_docs[0] and retrieved_docs[0].page_content == "doc1"
    assert retrieved_docs[1] and retrieved_docs[1].page_content == "doc2"


def test_yield_keys(mongo_store: MongoDBStore) -> None:
    mongo_store.mset(
        [
            ("key1", Document(page_content="doc1")),
            ("key2", Document(page_content="doc2")),
            ("another_key", Document(page_content="other")),
        ]
    )

    # Test without prefix
    keys = list(mongo_store.yield_keys())
    assert set(keys) == {"key1", "key2", "another_key"}

    # Test with prefix
    keys_with_prefix = list(mongo_store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2"}


def test_mdelete(mongo_store: MongoDBStore) -> None:
    mongo_store.mset(
        [
            ("key1", Document(page_content="doc1")),
            ("key2", Document(page_content="doc2")),
        ]
    )
    # Delete single document
    mongo_store.mdelete(["key1"])
    remaining_docs = list(mongo_store.yield_keys())
    assert "key1" not in remaining_docs
    assert "key2" in remaining_docs

    # Delete multiple documents
    mongo_store.mdelete(["key2"])
    remaining_docs = list(mongo_store.yield_keys())
    assert len(remaining_docs) == 0


def test_init_errors() -> None:
    with pytest.raises(ValueError):
        MongoDBStore("", "", "")


@pytest.fixture
def mongo_bytes_store() -> Generator:
    import mongomock

    # mongomock creates a mock MongoDB instance for testing purposes
    with mongomock.patch(servers=(("localhost", 27017),)):
        yield MongoDBByteStore(
            "mongodb://localhost:27017/", "test_db", "test_collection"
        )


def test_bytes_mset_and_mget(mongo_bytes_store: MongoDBByteStore) -> None:
    bytes1 = "doc1".encode("utf-8")
    bytes2 = "doc2".encode("utf-8")

    # Set documents in the store
    mongo_bytes_store.mset([("key1", bytes1), ("key2", bytes2)])

    # Get documents from the store
    retrieved_bytes = mongo_bytes_store.mget(["key1", "key2"])

    assert retrieved_bytes[0] and retrieved_bytes[0] == bytes1
    assert retrieved_bytes[1] and retrieved_bytes[1] == bytes2


def test_bytes_yield_keys(mongo_bytes_store: MongoDBByteStore) -> None:
    mongo_bytes_store.mset(
        [
            ("key1", "doc1".encode("utf-8")),
            ("key2", "doc2".encode("utf-8")),
            ("another_key", "other".encode("utf-8")),
        ]
    )

    # Test without prefix
    keys = list(mongo_bytes_store.yield_keys())
    assert set(keys) == {"key1", "key2", "another_key"}

    # Test with prefix
    keys_with_prefix = list(mongo_bytes_store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2"}


def test_bytes_mdelete(mongo_bytes_store: MongoDBByteStore) -> None:
    mongo_bytes_store.mset(
        [
            ("key1", "doc1".encode("utf-8")),
            ("key2", "doc2".encode("utf-8")),
        ]
    )
    # Delete single document
    mongo_bytes_store.mdelete(["key1"])
    remaining_docs = list(mongo_bytes_store.yield_keys())
    assert "key1" not in remaining_docs
    assert "key2" in remaining_docs

    # Delete multiple documents
    mongo_bytes_store.mdelete(["key2"])
    remaining_docs = list(mongo_bytes_store.yield_keys())
    assert len(remaining_docs) == 0


def test_bytes_init_errors() -> None:
    with pytest.raises(ValueError):
        MongoDBByteStore("", "", "")
