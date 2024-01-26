"""Implement integration tests for SQL storage."""
import os

from langchain_core.documents import Document

from langchain_community.storage.sql import SQLDocStore, SQLStrStore


def connection_string_from_db_params() -> str:
    """Return connection string from database parameters."""
    dbdriver = os.environ.get("TEST_SQL_DBDRIVER", "postgresql+psycopg2")
    host = os.environ.get("TEST_SQL_HOST", "localhost")
    port = int(os.environ.get("TEST_SQL_PORT", "5432"))
    database = os.environ.get("TEST_SQL_DATABASE", "postgres")
    user = os.environ.get("TEST_SQL_USER", "postgres")
    password = os.environ.get("TEST_SQL_PASSWORD", "postgres")
    return f"{dbdriver}://{user}:{password}@{host}:{port}/{database}"


CONNECTION_STRING = connection_string_from_db_params()
COLLECTION_NAME = "test_collection"
COLLECTION_NAME_2 = "test_collection_2"


def test_str_store_mget() -> None:
    store = SQLStrStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]

    # Test non-existent key
    non_existent_value = store.mget(["key3"])
    assert non_existent_value == [None]
    store.delete_collection()


def test_str_store_mset() -> None:
    store = SQLStrStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]
    store.delete_collection()


def test_str_store_mdelete() -> None:
    store = SQLStrStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store.mset([("key1", "value1"), ("key2", "value2")])

    store.mdelete(["key1"])

    values = store.mget(["key1", "key2"])
    assert values == [None, "value2"]

    # Test deleting non-existent key
    store.mdelete(["key3"])  # No error should be raised
    store.delete_collection()


def test_str_store_yield_keys() -> None:
    store = SQLStrStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store.mset([("key1", "value1"), ("key2", "value2"), ("key3", "value3")])

    keys = list(store.yield_keys())
    assert set(keys) == {"key1", "key2", "key3"}

    keys_with_prefix = list(store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2", "key3"}

    keys_with_invalid_prefix = list(store.yield_keys(prefix="x"))
    assert keys_with_invalid_prefix == []
    store.delete_collection()


def test_str_store_collection() -> None:
    """Test that collections are isolated within a db."""
    store_1 = SQLStrStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store_2 = SQLStrStore(
        collection_name=COLLECTION_NAME_2,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    store_1.mset([("key1", "value1"), ("key2", "value2")])
    store_2.mset([("key3", "value3"), ("key4", "value4")])

    values = store_1.mget(["key1", "key2"])
    assert values == ["value1", "value2"]
    values = store_1.mget(["key3", "key4"])
    assert values == [None, None]

    values = store_2.mget(["key1", "key2"])
    assert values == [None, None]
    values = store_2.mget(["key3", "key4"])
    assert values == ["value3", "value4"]

    store_1.delete_collection()
    store_2.delete_collection()


def test_doc_store_mget() -> None:
    store = SQLDocStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    doc_1 = Document(page_content="value1")
    doc_2 = Document(page_content="value2")
    store.mset([("key1", doc_1), ("key2", doc_2)])

    values = store.mget(["key1", "key2"])
    assert values == [doc_1, doc_2]

    # Test non-existent key
    non_existent_value = store.mget(["key3"])
    assert non_existent_value == [None]
    store.delete_collection()


def test_doc_store_mset() -> None:
    store = SQLDocStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    doc_1 = Document(page_content="value1")
    doc_2 = Document(page_content="value2")
    store.mset([("key1", doc_1), ("key2", doc_2)])

    values = store.mget(["key1", "key2"])
    assert values == [doc_1, doc_2]
    store.delete_collection()


def test_doc_store_mdelete() -> None:
    store = SQLDocStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    doc_1 = Document(page_content="value1")
    doc_2 = Document(page_content="value2")
    store.mset([("key1", doc_1), ("key2", doc_2)])

    store.mdelete(["key1"])

    values = store.mget(["key1", "key2"])
    assert values == [None, doc_2]

    # Test deleting non-existent key
    store.mdelete(["key3"])  # No error should be raised
    store.delete_collection()


def test_doc_store_yield_keys() -> None:
    store = SQLDocStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    doc_1 = Document(page_content="value1")
    doc_2 = Document(page_content="value2")
    doc_3 = Document(page_content="value3")
    store.mset([("key1", doc_1), ("key2", doc_2), ("key3", doc_3)])

    keys = list(store.yield_keys())
    assert set(keys) == {"key1", "key2", "key3"}

    keys_with_prefix = list(store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2", "key3"}

    keys_with_invalid_prefix = list(store.yield_keys(prefix="x"))
    assert keys_with_invalid_prefix == []
    store.delete_collection()


def test_doc_store_collection() -> None:
    """Test that collections are isolated within a db."""
    store_1 = SQLDocStore(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    store_2 = SQLDocStore(
        collection_name=COLLECTION_NAME_2,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    doc_1 = Document(page_content="value1")
    doc_2 = Document(page_content="value2")
    doc_3 = Document(page_content="value3")
    doc_4 = Document(page_content="value4")
    store_1.mset([("key1", doc_1), ("key2", doc_2)])
    store_2.mset([("key3", doc_3), ("key4", doc_4)])

    values = store_1.mget(["key1", "key2"])
    assert values == [doc_1, doc_2]
    values = store_1.mget(["key3", "key4"])
    assert values == [None, None]

    values = store_2.mget(["key1", "key2"])
    assert values == [None, None]
    values = store_2.mget(["key3", "key4"])
    assert values == [doc_3, doc_4]

    store_1.delete_collection()
    store_2.delete_collection()
