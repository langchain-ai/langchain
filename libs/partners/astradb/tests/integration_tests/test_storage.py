"""Implement integration tests for AstraDB storage."""
from __future__ import annotations

import os

import pytest
from astrapy.db import AstraDB, AsyncAstraDB

from langchain_astradb.storage import AstraDBByteStore, AstraDBStore
from langchain_astradb.utils.astradb import SetupMode


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


@pytest.fixture
def astra_db() -> AstraDB:
    from astrapy.db import AstraDB

    return AstraDB(
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


@pytest.fixture
def async_astra_db() -> AsyncAstraDB:
    from astrapy.db import AsyncAstraDB

    return AsyncAstraDB(
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


def init_store(astra_db: AstraDB, collection_name: str) -> AstraDBStore:
    store = AstraDBStore(collection_name=collection_name, astra_db_client=astra_db)
    store.mset([("key1", [0.1, 0.2]), ("key2", "value2")])
    return store


def init_bytestore(astra_db: AstraDB, collection_name: str) -> AstraDBByteStore:
    store = AstraDBByteStore(collection_name=collection_name, astra_db_client=astra_db)
    store.mset([("key1", b"value1"), ("key2", b"value2")])
    return store


async def init_async_store(
    async_astra_db: AsyncAstraDB, collection_name: str
) -> AstraDBStore:
    store = AstraDBStore(
        collection_name=collection_name,
        async_astra_db_client=async_astra_db,
        setup_mode=SetupMode.ASYNC,
    )
    await store.amset([("key1", [0.1, 0.2]), ("key2", "value2")])
    return store


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBStore:
    def test_mget(self, astra_db: AstraDB) -> None:
        """Test AstraDBStore mget method."""
        collection_name = "lc_test_store_mget"
        try:
            store = init_store(astra_db, collection_name)
            assert store.mget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        finally:
            astra_db.delete_collection(collection_name)

    async def test_amget(self, async_astra_db: AsyncAstraDB) -> None:
        """Test AstraDBStore amget method."""
        collection_name = "lc_test_store_mget"
        try:
            store = await init_async_store(async_astra_db, collection_name)
            assert await store.amget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        finally:
            await async_astra_db.delete_collection(collection_name)

    def test_mset(self, astra_db: AstraDB) -> None:
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

    async def test_amset(self, async_astra_db: AsyncAstraDB) -> None:
        """Test that multiple keys can be set with AstraDBStore."""
        collection_name = "lc_test_store_mset"
        try:
            store = await init_async_store(async_astra_db, collection_name)
            result = await store.async_collection.find_one({"_id": "key1"})
            assert result["data"]["document"]["value"] == [0.1, 0.2]
            result = await store.async_collection.find_one({"_id": "key2"})
            assert result["data"]["document"]["value"] == "value2"
        finally:
            await async_astra_db.delete_collection(collection_name)

    def test_mdelete(self, astra_db: AstraDB) -> None:
        """Test that deletion works as expected."""
        collection_name = "lc_test_store_mdelete"
        try:
            store = init_store(astra_db, collection_name)
            store.mdelete(["key1", "key2"])
            result = store.mget(["key1", "key2"])
            assert result == [None, None]
        finally:
            astra_db.delete_collection(collection_name)

    async def test_amdelete(self, async_astra_db: AsyncAstraDB) -> None:
        """Test that deletion works as expected."""
        collection_name = "lc_test_store_mdelete"
        try:
            store = await init_async_store(async_astra_db, collection_name)
            await store.amdelete(["key1", "key2"])
            result = await store.amget(["key1", "key2"])
            assert result == [None, None]
        finally:
            await async_astra_db.delete_collection(collection_name)

    def test_yield_keys(self, astra_db: AstraDB) -> None:
        collection_name = "lc_test_store_yield_keys"
        try:
            store = init_store(astra_db, collection_name)
            assert set(store.yield_keys()) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="key")) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="lang")) == set()
        finally:
            astra_db.delete_collection(collection_name)

    async def test_ayield_keys(self, async_astra_db: AsyncAstraDB) -> None:
        collection_name = "lc_test_store_yield_keys"
        try:
            store = await init_async_store(async_astra_db, collection_name)
            assert {key async for key in store.ayield_keys()} == {"key1", "key2"}
            assert {key async for key in store.ayield_keys(prefix="key")} == {
                "key1",
                "key2",
            }
            assert {key async for key in store.ayield_keys(prefix="lang")} == set()
        finally:
            await async_astra_db.delete_collection(collection_name)

    def test_bytestore_mget(self, astra_db: AstraDB) -> None:
        """Test AstraDBByteStore mget method."""
        collection_name = "lc_test_bytestore_mget"
        try:
            store = init_bytestore(astra_db, collection_name)
            assert store.mget(["key1", "key2"]) == [b"value1", b"value2"]
        finally:
            astra_db.delete_collection(collection_name)

    def test_bytestore_mset(self, astra_db: AstraDB) -> None:
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
