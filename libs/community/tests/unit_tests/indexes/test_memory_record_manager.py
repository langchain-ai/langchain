import pytest
import pytest_asyncio

from langchain_community.indexes.memory_recordmanager import MemoryRecordManager


@pytest.fixture()
def manager() -> MemoryRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = MemoryRecordManager(namespace="kittens")
    return record_manager


@pytest_asyncio.fixture  # type: ignore
async def amanager() -> MemoryRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = MemoryRecordManager(namespace="kittens")
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
