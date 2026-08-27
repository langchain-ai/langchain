from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio

from langchain_core.indexing import InMemoryRecordManager


@pytest.fixture
def manager() -> InMemoryRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = InMemoryRecordManager(namespace="kittens")
    record_manager.create_schema()
    return record_manager


@pytest_asyncio.fixture()
async def amanager() -> InMemoryRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = InMemoryRecordManager(namespace="kittens")
    await record_manager.acreate_schema()
    return record_manager


def test_update(manager: InMemoryRecordManager) -> None:
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


async def test_aupdate(amanager: InMemoryRecordManager) -> None:
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


def test_update_timestamp(manager: InMemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        manager.update(["key1"])

    assert manager.list_keys() == ["key1"]
    assert (
        manager.list_keys(before=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
        == []
    )
    assert manager.list_keys(
        after=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]
    assert (
        manager.list_keys(after=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp())
        == []
    )

    # Update the timestamp
    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2023, 1, 5, tzinfo=timezone.utc).timestamp(),
    ):
        manager.update(["key1"])

    assert manager.list_keys() == ["key1"]
    assert (
        manager.list_keys(before=datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
        == []
    )
    assert manager.list_keys(
        after=datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]
    assert manager.list_keys(
        after=datetime(2023, 1, 3, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]


async def test_aupdate_timestamp(manager: InMemoryRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        await manager.aupdate(["key1"])

    assert await manager.alist_keys() == ["key1"]
    assert (
        await manager.alist_keys(
            before=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
        )
        == []
    )
    assert await manager.alist_keys(
        after=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]
    assert (
        await manager.alist_keys(
            after=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp()
        )
        == []
    )

    # Update the timestamp
    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2023, 1, 5, tzinfo=timezone.utc).timestamp(),
    ):
        await manager.aupdate(["key1"])

    assert await manager.alist_keys() == ["key1"]
    assert (
        await manager.alist_keys(
            before=datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
        )
        == []
    )
    assert await manager.alist_keys(
        after=datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]
    assert await manager.alist_keys(
        after=datetime(2023, 1, 3, tzinfo=timezone.utc).timestamp()
    ) == ["key1"]


def test_exists(manager: InMemoryRecordManager) -> None:
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


async def test_aexists(amanager: InMemoryRecordManager) -> None:
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


async def test_list_keys(manager: InMemoryRecordManager) -> None:
    """Test listing keys based on the provided date range."""
    # Insert records
    assert manager.list_keys() == []
    assert await manager.alist_keys() == []

    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        manager.update(["key1", "key2"])
        manager.update(["key3"], group_ids=["group1"])
        manager.update(["key4"], group_ids=["group2"])

    with patch.object(
        manager,
        "get_time",
        return_value=datetime(2021, 1, 10, tzinfo=timezone.utc).timestamp(),
    ):
        manager.update(["key5"])

    assert sorted(manager.list_keys()) == ["key1", "key2", "key3", "key4", "key5"]
    assert sorted(await manager.alist_keys()) == [
        "key1",
        "key2",
        "key3",
        "key4",
        "key5",
    ]

    # By group
    assert manager.list_keys(group_ids=["group1"]) == ["key3"]
    assert await manager.alist_keys(group_ids=["group1"]) == ["key3"]

    # Before
    assert sorted(
        manager.list_keys(before=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp())
    ) == [
        "key1",
        "key2",
        "key3",
        "key4",
    ]
    assert sorted(
        await manager.alist_keys(
            before=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp()
        )
    ) == [
        "key1",
        "key2",
        "key3",
        "key4",
    ]

    # After
    assert sorted(
        manager.list_keys(after=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp())
    ) == ["key5"]
    assert sorted(
        await manager.alist_keys(
            after=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp()
        )
    ) == ["key5"]

    results = manager.list_keys(limit=1)
    assert len(results) == 1
    assert results[0] in {"key1", "key2", "key3", "key4", "key5"}

    results = await manager.alist_keys(limit=1)
    assert len(results) == 1
    assert results[0] in {"key1", "key2", "key3", "key4", "key5"}


def test_delete_keys(manager: InMemoryRecordManager) -> None:
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


async def test_adelete_keys(amanager: InMemoryRecordManager) -> None:
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
