from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from langchain_community.indexes._memory_recordmanager import MemoryRecordManager


@pytest.fixture()
def manager() -> MemoryRecordManager:
    """Initialize the test memory database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = MemoryRecordManager("kittens")
    record_manager.create_schema()
    return record_manager


@pytest_asyncio.fixture  # type: ignore
@pytest.mark.requires("aiosqlite")
async def amanager() -> MemoryRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = MemoryRecordManager("kittens")
    await record_manager.acreate_schema()
    return record_manager


def test_update(manager: MemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = manager.list_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    manager.update(keys)
    # Retrieve the records
    read_keys = manager.list_keys()
    assert read_keys == ["key1", "key2", "key3"]


@pytest.mark.requires("aiosqlite")
async def test_aupdate(amanager: MemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = await amanager.alist_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    await amanager.aupdate(keys)
    # Retrieve the records
    read_keys = await amanager.alist_keys()
    assert read_keys == ["key1", "key2", "key3"]


def test_update_with_group_ids(manager: MemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = manager.list_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    manager.update(keys)
    # Retrieve the records
    read_keys = manager.list_keys()
    assert read_keys == ["key1", "key2", "key3"]


@pytest.mark.requires("aiosqlite")
async def test_aupdate_with_group_ids(amanager: MemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = await amanager.alist_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    await amanager.aupdate(keys)
    # Retrieve the records
    read_keys = await amanager.alist_keys()
    assert read_keys == ["key1", "key2", "key3"]


def test_exists(manager: MemoryRecordManager) -> None:
    """Test checking if keys exist in the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    manager.update(keys)
    # Check if the keys exist in the database
    exists = manager.exists(keys)
    assert len(exists) == len(keys)
    assert exists == [True, True, True]

    exists = manager.exists(["key1", "key4"])
    assert len(exists) == 2
    assert exists == [True, False]


@pytest.mark.requires("aiosqlite")
async def test_aexists(amanager: MemoryRecordManager) -> None:
    """Test checking if keys exist in the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    await amanager.aupdate(keys)
    # Check if the keys exist in the database
    exists = await amanager.aexists(keys)
    assert len(exists) == len(keys)
    assert exists == [True, True, True]

    exists = await amanager.aexists(["key1", "key4"])
    assert len(exists) == 2
    assert exists == [True, False]


def test_list_keys(manager: MemoryRecordManager) -> None:
    """Test listing keys based on the provided date range."""
    # Insert records
    assert manager.list_keys() == []
    manager.data=[
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key2", "updated_at": datetime(2022, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key3", "updated_at": datetime(2023, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key4", "updated_at": datetime(2024, 1, 1).timestamp(), "group_id": "group1", "namespace": "kittens"},
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
        {"key": "key5", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
    ]

    # Retrieve all keys
    assert manager.list_keys() == ["key1", "key2", "key3", "key4"]

    # Retrieve keys updated after a certain date
    assert manager.list_keys(after=datetime(2022, 2, 1).timestamp()) == ["key3", "key4"]

    # Retrieve keys updated after a certain date
    assert manager.list_keys(before=datetime(2022, 2, 1).timestamp()) == [
        "key1",
        "key2",
    ]

    # Retrieve keys updated after a certain date
    assert manager.list_keys(before=datetime(2019, 2, 1).timestamp()) == []

    # Retrieve keys in a time range
    assert manager.list_keys(
        before=datetime(2022, 2, 1).timestamp(),
        after=datetime(2021, 11, 1).timestamp(),
    ) == ["key2"]

    assert manager.list_keys(group_ids=["group1", "group2"]) == ["key4"]

    # Test multiple filters
    assert (
        manager.list_keys(
            group_ids=["group1", "group2"], before=datetime(2019, 1, 1).timestamp()
        )
        == []
    )
    assert manager.list_keys(
        group_ids=["group1", "group2"], after=datetime(2019, 1, 1).timestamp()
    ) == ["key4"]


@pytest.mark.requires("aiosqlite")
async def test_list_keys(manager: MemoryRecordManager) -> None:
    """Test listing keys based on the provided date range."""
    # Insert records
    assert (await manager.alist_keys()) == []
    manager.data=[
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key2", "updated_at": datetime(2022, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key3", "updated_at": datetime(2023, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key4", "updated_at": datetime(2024, 1, 1).timestamp(), "group_id": "group1", "namespace": "kittens"},
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
        {"key": "key5", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
    ]

    # Retrieve all keys
    assert await manager.alist_keys() == ["key1", "key2", "key3", "key4"]

    # Retrieve keys updated after a certain date
    assert await manager.alist_keys(after=datetime(2022, 2, 1).timestamp()) == ["key3", "key4"]

    # Retrieve keys updated after a certain date
    assert await manager.alist_keys(before=datetime(2022, 2, 1).timestamp()) == [
        "key1",
        "key2",
    ]

    # Retrieve keys updated after a certain date
    assert await manager.alist_keys(before=datetime(2019, 2, 1).timestamp()) == []

    # Retrieve keys in a time range
    assert await manager.alist_keys(
        before=datetime(2022, 2, 1).timestamp(),
        after=datetime(2021, 11, 1).timestamp(),
    ) == ["key2"]

    assert await manager.alist_keys(group_ids=["group1", "group2"]) == ["key4"]

    # Test multiple filters
    assert (
        await manager.alist_keys(
            group_ids=["group1", "group2"], before=datetime(2019, 1, 1).timestamp()
        )
        == []
    )
    assert await manager.alist_keys(
        group_ids=["group1", "group2"], after=datetime(2019, 1, 1).timestamp()
    ) == ["key4"]


def test_namespace_is_used(manager: MemoryRecordManager) -> None:
    """Verify that namespace is taken into account for all operations."""
    assert manager.namespace == "kittens"
    manager.data=[
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key2", "updated_at": datetime(2022, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
        {"key": "key3", "updated_at": datetime(2023, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
    ]

    assert manager.list_keys() == ["key1", "key2"]
    manager.delete_keys(["key1"])
    assert manager.list_keys() == ["key2"]
    manager.update(["key3"], group_ids=["group3"])

    assert sorted([(r['namespace'], r['key'], r['group_id']) for r in manager.data]) == [
        ("kittens", "key2", None),
        ("kittens", "key3", "group3"),
        ("puppies", "key1", None),
        ("puppies", "key3", None),
    ]


@pytest.mark.requires("aiosqlite")
async def test_namespace_is_used(manager: MemoryRecordManager) -> None:
    """Verify that namespace is taken into account for all operations."""
    assert manager.namespace == "kittens"
    manager.data=[
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key2", "updated_at": datetime(2022, 1, 1).timestamp(), "group_id": None, "namespace": "kittens"},
        {"key": "key1", "updated_at": datetime(2021, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
        {"key": "key3", "updated_at": datetime(2023, 1, 1).timestamp(), "group_id": None, "namespace": "puppies"},
    ]

    assert await manager.alist_keys() == ["key1", "key2"]
    await manager.adelete_keys(["key1"])
    assert await manager.alist_keys() == ["key2"]
    await manager.aupdate(["key3"], group_ids=["group3"])

    assert sorted([(r['namespace'], r['key'], r['group_id']) for r in manager.data]) == [
        ("kittens", "key2", None),
        ("kittens", "key3", "group3"),
        ("puppies", "key1", None),
        ("puppies", "key3", None),
    ]




def test_delete_keys(manager: MemoryRecordManager) -> None:
    """Test deleting keys from the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    manager.update(keys)

    # Delete some keys
    keys_to_delete = ["key1", "key2"]
    manager.delete_keys(keys_to_delete)

    # Check if the deleted keys are no longer in the database
    remaining_keys = manager.list_keys()
    assert remaining_keys == ["key3"]


@pytest.mark.requires("aiosqlite")
async def test_adelete_keys(amanager: MemoryRecordManager) -> None:
    """Test deleting keys from the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    await amanager.aupdate(keys)

    # Delete some keys
    keys_to_delete = ["key1", "key2"]
    await amanager.adelete_keys(keys_to_delete)

    # Check if the deleted keys are no longer in the database
    remaining_keys = await amanager.alist_keys()
    assert remaining_keys == ["key3"]
