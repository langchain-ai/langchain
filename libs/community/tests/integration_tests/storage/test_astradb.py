"""Implement integration tests for AstraDB storage."""
import os

import pytest

from langchain_community.storage.astradb import AstraDBByteStore, AstraDBStore


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


@pytest.fixture
def astra_db():
    from astrapy.db import AstraDB

    return AstraDB(
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


def init_store(astra_db, collection_name: str):
    astra_db.create_collection(collection_name)
    store = AstraDBStore(collection_name=collection_name, astra_db_client=astra_db)
    store.mset([("key1", [0.1, 0.2]), ("key2", "value2")])
    return store


def init_bytestore(astra_db, collection_name: str):
    astra_db.create_collection(collection_name)
    store = AstraDBByteStore(collection_name=collection_name, astra_db_client=astra_db)
    store.mset([("key1", b"value1"), ("key2", b"value2")])
    return store


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBStore:
    def test_mget(self, astra_db) -> None:
        """Test AstraDBStore mget method."""
        collection_name = "lc_test_store_mget"
        try:
            store = init_store(astra_db, collection_name)
            assert store.mget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        finally:
            astra_db.delete_collection(collection_name)

    def test_mset(self, astra_db) -> None:
        """Test that multiple keys can be set with AstraDBStore."""
        collection_name = "lc_test_store_mset"
        try:
            store = init_store(astra_db, collection_name)
            result = store.collection.find_one({"_id": "key1"})
            assert result["data"]["document"]["value"] == [0.1, 0.2]
            result = store.collection.find_one({"_id": "key2"})
            assert result["data"]["document"]["value"] == "value2"
        finally:
            astra_db.delete_collection(collection_name)

    def test_mdelete(self, astra_db) -> None:
        """Test that deletion works as expected."""
        collection_name = "lc_test_store_mdelete"
        try:
            store = init_store(astra_db, collection_name)
            store.mdelete(["key1", "key2"])
            result = store.mget(["key1", "key2"])
            assert result == [None, None]
        finally:
            astra_db.delete_collection(collection_name)

    def test_yield_keys(self, astra_db) -> None:
        collection_name = "lc_test_store_yield_keys"
        try:
            store = init_store(astra_db, collection_name)
            assert set(store.yield_keys()) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="key")) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="lang")) == set()
        finally:
            astra_db.delete_collection(collection_name)

    def test_bytestore_mget(self, astra_db) -> None:
        """Test AstraDBByteStore mget method."""
        collection_name = "lc_test_bytestore_mget"
        try:
            store = init_bytestore(astra_db, collection_name)
            assert store.mget(["key1", "key2"]) == [b"value1", b"value2"]
        finally:
            astra_db.delete_collection(collection_name)

    def test_bytestore_mset(self, astra_db) -> None:
        """Test that multiple keys can be set with AstraDBByteStore."""
        collection_name = "lc_test_bytestore_mset"
        try:
            store = init_bytestore(astra_db, collection_name)
            result = store.collection.find_one({"_id": "key1"})
            assert result["data"]["document"]["value"] == "dmFsdWUx"
            result = store.collection.find_one({"_id": "key2"})
            assert result["data"]["document"]["value"] == "dmFsdWUy"
        finally:
            astra_db.delete_collection(collection_name)
