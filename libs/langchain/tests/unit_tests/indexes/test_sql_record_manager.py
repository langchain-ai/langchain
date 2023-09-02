from datetime import datetime
from unittest.mock import patch

import pytest

from langchain.indexes._sql_record_manager import SQLRecordManager, UpsertionRecord


@pytest.fixture()
def manager() -> SQLRecordManager:
    """Initialize the test database and yield the TimestampedSet instance."""
    # Initialize and yield the TimestampedSet instance
    record_manager = SQLRecordManager("kittens", db_url="sqlite:///:memory:")
    record_manager.create_schema()
    return record_manager


def test_update(manager: SQLRecordManager) -> None:
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


def test_update_timestamp(manager: SQLRecordManager) -> None:
    """Test updating records in the database."""
    # no keys should be present in the set
    with patch.object(
        manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        manager.update(["key1"])

    with manager._make_session() as session:
        records = (
            session.query(UpsertionRecord)
            .filter(UpsertionRecord.namespace == manager.namespace)
            .all()  # type: ignore[attr-defined]
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
        manager.update(["key1"])

    with manager._make_session() as session:
        records = (
            session.query(UpsertionRecord)
            .filter(UpsertionRecord.namespace == manager.namespace)
            .all()  # type: ignore[attr-defined]
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
        manager.update(["key1"], group_ids=["group1"])

    with manager._make_session() as session:
        records = (
            session.query(UpsertionRecord)
            .filter(UpsertionRecord.namespace == manager.namespace)
            .all()  # type: ignore[attr-defined]
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


def test_update_with_group_ids(manager: SQLRecordManager) -> None:
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


def test_exists(manager: SQLRecordManager) -> None:
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


def test_list_keys(manager: SQLRecordManager) -> None:
    """Test listing keys based on the provided date range."""
    # Insert records
    assert manager.list_keys() == []
    with manager._make_session() as session:
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
        session.commit()

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


def test_namespace_is_used(manager: SQLRecordManager) -> None:
    """Verify that namespace is taken into account for all operations."""
    assert manager.namespace == "kittens"
    with manager._make_session() as session:
        # Add some keys with explicit updated_ats
        session.add(UpsertionRecord(key="key1", namespace="kittens"))
        session.add(UpsertionRecord(key="key2", namespace="kittens"))
        session.add(UpsertionRecord(key="key1", namespace="puppies"))
        session.add(UpsertionRecord(key="key3", namespace="puppies"))
        session.commit()

    assert manager.list_keys() == ["key1", "key2"]
    manager.delete_keys(["key1"])
    assert manager.list_keys() == ["key2"]
    manager.update(["key3"], group_ids=["group3"])

    with manager._make_session() as session:
        results = session.query(UpsertionRecord).all()

        assert sorted([(r.namespace, r.key, r.group_id) for r in results]) == [
            ("kittens", "key2", None),
            ("kittens", "key3", "group3"),
            ("puppies", "key1", None),
            ("puppies", "key3", None),
        ]


def test_delete_keys(manager: SQLRecordManager) -> None:
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
