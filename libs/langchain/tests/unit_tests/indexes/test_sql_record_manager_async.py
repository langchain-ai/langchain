from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from langchain.indexes._sql_record_manager_async import (
    SQLRecordManagerAsync,
    UpsertionRecord,
)


@pytest_asyncio.fixture  # type: ignore
async def manager() -> SQLRecordManagerAsync:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = SQLRecordManagerAsync(
        "kittens",
        db_url="sqlite+aiosqlite:///:memory:",
    )
    await record_manager.create_schema()
    return record_manager


@pytest.mark.asyncio
async def test_update(manager: SQLRecordManagerAsync) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = await manager.list_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    await manager.update(keys)
    # Retrieve the records
    read_keys = await manager.list_keys()
    assert read_keys == ["key1", "key2", "key3"]


@pytest.mark.asyncio
async def test_update_timestamp(manager: SQLRecordManagerAsync) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    with patch.object(
        manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        await manager.update(["key1"])

    async with manager._make_session() as session:
        records = (
            (
                await session.execute(
                    select(UpsertionRecord).filter(
                        UpsertionRecord.namespace == manager.namespace
                    )
                )
            )
            .scalars()
            .all()
        )

        assert [
            {
                "key": record.key,
                "namespace": record.namespace,
                "updated_at": record.updated_at,
                "group_id": record.group_id,
            }
            for record in records
        ] == [
            {
                "group_id": None,
                "key": "key1",
                "namespace": "kittens",
                "updated_at": datetime(2021, 1, 2, 0, 0).timestamp(),
            }
        ]

    with patch.object(
        manager, "get_time", return_value=datetime(2023, 1, 2).timestamp()
    ):
        await manager.update(["key1"])

    async with manager._make_session() as session:
        records = (
            (
                await session.execute(
                    select(UpsertionRecord).filter(
                        UpsertionRecord.namespace == manager.namespace
                    )
                )
            )
            .scalars()
            .all()
        )

        assert [
            {
                "key": record.key,
                "namespace": record.namespace,
                "updated_at": record.updated_at,
                "group_id": record.group_id,
            }
            for record in records
        ] == [
            {
                "group_id": None,
                "key": "key1",
                "namespace": "kittens",
                "updated_at": datetime(2023, 1, 2, 0, 0).timestamp(),
            }
        ]

    with patch.object(
        manager, "get_time", return_value=datetime(2023, 2, 2).timestamp()
    ):
        await manager.update(["key1"], group_ids=["group1"])

    async with manager._make_session() as session:
        records = (
            (
                await session.execute(
                    select(UpsertionRecord).filter(
                        UpsertionRecord.namespace == manager.namespace
                    )
                )
            )
            .scalars()
            .all()
        )

        assert [
            {
                "key": record.key,
                "namespace": record.namespace,
                "updated_at": record.updated_at,
                "group_id": record.group_id,
            }
            for record in records
        ] == [
            {
                "group_id": "group1",
                "key": "key1",
                "namespace": "kittens",
                "updated_at": datetime(2023, 2, 2, 0, 0).timestamp(),
            }
        ]


@pytest.mark.asyncio
async def test_update_with_group_ids(manager: SQLRecordManagerAsync) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    read_keys = await manager.list_keys()
    assert read_keys == []
    # Insert records
    keys = ["key1", "key2", "key3"]
    await manager.update(keys)
    # Retrieve the records
    read_keys = await manager.list_keys()
    assert read_keys == ["key1", "key2", "key3"]


@pytest.mark.asyncio
async def test_exists(manager: SQLRecordManagerAsync) -> None:
    """Test checking if keys exist in the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    await manager.update(keys)
    # Check if the keys exist in the database
    exists = await manager.exists(keys)
    assert len(exists) == len(keys)
    assert exists == [True, True, True]

    exists = await manager.exists(["key1", "key4"])
    assert len(exists) == 2
    assert exists == [True, False]


@pytest.mark.asyncio
async def test_list_keys(manager: SQLRecordManagerAsync) -> None:
    """Test listing keys based on the provided date range."""
    # Insert records
    assert await manager.list_keys() == []
    async with manager._make_session() as session:
        # Add some keys with explicit updated_ats
        session.add(
            UpsertionRecord(
                key="key1",
                updated_at=datetime(2021, 1, 1).timestamp(),
                namespace="kittens",
            )
        )
        session.add(
            UpsertionRecord(
                key="key2",
                updated_at=datetime(2022, 1, 1).timestamp(),
                namespace="kittens",
            )
        )
        session.add(
            UpsertionRecord(
                key="key3",
                updated_at=datetime(2023, 1, 1).timestamp(),
                namespace="kittens",
            )
        )
        session.add(
            UpsertionRecord(
                key="key4",
                group_id="group1",
                updated_at=datetime(2024, 1, 1).timestamp(),
                namespace="kittens",
            )
        )
        # Insert keys from a different namespace, these should not be visible!
        session.add(
            UpsertionRecord(
                key="key1",
                updated_at=datetime(2021, 1, 1).timestamp(),
                namespace="puppies",
            )
        )
        session.add(
            UpsertionRecord(
                key="key5",
                updated_at=datetime(2021, 1, 1).timestamp(),
                namespace="puppies",
            )
        )
        await session.commit()

    # Retrieve all keys
    assert await manager.list_keys() == ["key1", "key2", "key3", "key4"]

    # Retrieve keys updated after a certain date
    assert await manager.list_keys(after=datetime(2022, 2, 1).timestamp()) == [
        "key3",
        "key4",
    ]

    # Retrieve keys updated after a certain date
    assert await manager.list_keys(before=datetime(2022, 2, 1).timestamp()) == [
        "key1",
        "key2",
    ]

    # Retrieve keys updated after a certain date
    assert await manager.list_keys(before=datetime(2019, 2, 1).timestamp()) == []

    # Retrieve keys in a time range
    assert await manager.list_keys(
        before=datetime(2022, 2, 1).timestamp(),
        after=datetime(2021, 11, 1).timestamp(),
    ) == ["key2"]

    assert await manager.list_keys(group_ids=["group1", "group2"]) == ["key4"]

    # Test multiple filters
    assert (
        await manager.list_keys(
            group_ids=["group1", "group2"], before=datetime(2019, 1, 1).timestamp()
        )
        == []
    )
    assert await manager.list_keys(
        group_ids=["group1", "group2"], after=datetime(2019, 1, 1).timestamp()
    ) == ["key4"]


@pytest.mark.asyncio
async def test_namespace_is_used(manager: SQLRecordManagerAsync) -> None:
    """Verify that namespace is taken into account for all operations."""
    assert manager.namespace == "kittens"
    async with manager._make_session() as session:
        # Add some keys with explicit updated_ats
        session.add(UpsertionRecord(key="key1", namespace="kittens"))
        session.add(UpsertionRecord(key="key2", namespace="kittens"))
        session.add(UpsertionRecord(key="key1", namespace="puppies"))
        session.add(UpsertionRecord(key="key3", namespace="puppies"))
        await session.commit()

    assert await manager.list_keys() == ["key1", "key2"]
    await manager.delete_keys(["key1"])
    assert await manager.list_keys() == ["key2"]
    await manager.update(["key3"], group_ids=["group3"])

    async with manager._make_session() as session:
        results = (await session.execute(select(UpsertionRecord))).scalars().all()

        assert sorted([(r.namespace, r.key, r.group_id) for r in results]) == [
            ("kittens", "key2", None),
            ("kittens", "key3", "group3"),
            ("puppies", "key1", None),
            ("puppies", "key3", None),
        ]


@pytest.mark.asyncio
async def test_delete_keys(manager: SQLRecordManagerAsync) -> None:
    """Test deleting keys from the database."""
    # Insert records
    keys = ["key1", "key2", "key3"]
    await manager.update(keys)

    # Delete some keys
    keys_to_delete = ["key1", "key2"]
    await manager.delete_keys(keys_to_delete)

    # Check if the deleted keys are no longer in the database
    remaining_keys = await manager.list_keys()
    assert remaining_keys == ["key3"]
