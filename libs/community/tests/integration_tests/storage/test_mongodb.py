from typing import Generator, Tuple

import pytest
from langchain_core.documents import Document
from langchain_standard_tests.integration_tests.base_store import BaseStoreSyncTests

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


class TestMongoDBStore(BaseStoreSyncTests):
    @pytest.fixture
    def three_values(self) -> Tuple[bytes, bytes, bytes]:  # <-- Provide 3
        return b"foo", b"bar", b"buzz"

    @pytest.fixture
    def kv_store(self) -> MongoDBByteStore:
        import mongomock

        # mongomock creates a mock MongoDB instance for testing purposes
        with mongomock.patch(servers=(("localhost", 27017),)):
            return MongoDBByteStore(
                "mongodb://localhost:27017/", "test_db", "test_collection"
            )
