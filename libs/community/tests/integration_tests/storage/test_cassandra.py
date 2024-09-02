"""Implement integration tests for Cassandra storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain_community.storage.cassandra import CassandraByteStore
from langchain_community.utilities.cassandra import SetupMode

if TYPE_CHECKING:
    from cassandra.cluster import Session

KEYSPACE = "storage_test_keyspace"


@pytest.fixture(scope="session")
def session() -> Session:
    from cassandra.cluster import Cluster

    cluster = Cluster()
    session = cluster.connect()
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    return session


def init_store(table_name: str, session: Session) -> CassandraByteStore:
    store = CassandraByteStore(table=table_name, keyspace=KEYSPACE, session=session)
    store.mset([("key1", b"value1"), ("key2", b"value2")])
    return store


async def init_async_store(table_name: str, session: Session) -> CassandraByteStore:
    store = CassandraByteStore(
        table=table_name, keyspace=KEYSPACE, session=session, setup_mode=SetupMode.ASYNC
    )
    await store.amset([("key1", b"value1"), ("key2", b"value2")])
    return store


def drop_table(table_name: str, session: Session) -> None:
    session.execute(f"DROP TABLE {KEYSPACE}.{table_name}")


async def test_mget(session: Session) -> None:
    """Test CassandraByteStore mget method."""
    table_name = "lc_test_store_mget"
    try:
        store = init_store(table_name, session)
        assert store.mget(["key1", "key2"]) == [b"value1", b"value2"]
        assert await store.amget(["key1", "key2"]) == [b"value1", b"value2"]
    finally:
        drop_table(table_name, session)


async def test_amget(session: Session) -> None:
    """Test CassandraByteStore amget method."""
    table_name = "lc_test_store_amget"
    try:
        store = await init_async_store(table_name, session)
        assert await store.amget(["key1", "key2"]) == [b"value1", b"value2"]
    finally:
        drop_table(table_name, session)


def test_mset(session: Session) -> None:
    """Test that multiple keys can be set with CassandraByteStore."""
    table_name = "lc_test_store_mset"
    try:
        init_store(table_name, session)
        result = session.execute(
            "SELECT row_id, body_blob FROM storage_test_keyspace.lc_test_store_mset "
            "WHERE row_id = 'key1';"
        ).one()
        assert result.body_blob == b"value1"
        result = session.execute(
            "SELECT row_id, body_blob FROM storage_test_keyspace.lc_test_store_mset "
            "WHERE row_id = 'key2';"
        ).one()
        assert result.body_blob == b"value2"
    finally:
        drop_table(table_name, session)


async def test_amset(session: Session) -> None:
    """Test that multiple keys can be set with CassandraByteStore."""
    table_name = "lc_test_store_amset"
    try:
        await init_async_store(table_name, session)
        result = session.execute(
            "SELECT row_id, body_blob FROM storage_test_keyspace.lc_test_store_amset "
            "WHERE row_id = 'key1';"
        ).one()
        assert result.body_blob == b"value1"
        result = session.execute(
            "SELECT row_id, body_blob FROM storage_test_keyspace.lc_test_store_amset "
            "WHERE row_id = 'key2';"
        ).one()
        assert result.body_blob == b"value2"
    finally:
        drop_table(table_name, session)


def test_mdelete(session: Session) -> None:
    """Test that deletion works as expected."""
    table_name = "lc_test_store_mdelete"
    try:
        store = init_store(table_name, session)
        store.mdelete(["key1", "key2"])
        result = store.mget(["key1", "key2"])
        assert result == [None, None]
    finally:
        drop_table(table_name, session)


async def test_amdelete(session: Session) -> None:
    """Test that deletion works as expected."""
    table_name = "lc_test_store_amdelete"
    try:
        store = await init_async_store(table_name, session)
        await store.amdelete(["key1", "key2"])
        result = await store.amget(["key1", "key2"])
        assert result == [None, None]
    finally:
        drop_table(table_name, session)


def test_yield_keys(session: Session) -> None:
    table_name = "lc_test_store_yield_keys"
    try:
        store = init_store(table_name, session)
        assert set(store.yield_keys()) == {"key1", "key2"}
        assert set(store.yield_keys(prefix="key")) == {"key1", "key2"}
        assert set(store.yield_keys(prefix="lang")) == set()
    finally:
        drop_table(table_name, session)


async def test_ayield_keys(session: Session) -> None:
    table_name = "lc_test_store_ayield_keys"
    try:
        store = await init_async_store(table_name, session)
        assert {key async for key in store.ayield_keys()} == {"key1", "key2"}
        assert {key async for key in store.ayield_keys(prefix="key")} == {
            "key1",
            "key2",
        }
        assert {key async for key in store.ayield_keys(prefix="lang")} == set()
    finally:
        drop_table(table_name, session)
