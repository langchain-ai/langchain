from datetime import datetime
from typing import (
    AsyncIterator,
    Generator,
    Iterator,
    Sequence,
)
from unittest.mock import patch

import pytest
from langchain.indexes import index
from langchain.indexes._api import _HashedDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from langchain_community.indexes._dynamodb_manager import (
    GROUP_ID_FIELD,
    IMPORT_BOTO3_ERROR,
    KEY_FIELD,
    NAMESPACE_FIELD,
    UPDATED_AT_FIELD,
    DynamoDBRecordManager,
)


@pytest.fixture
def table_name() -> str:
    return "test_langchain_records"


@pytest.fixture
def create_table(table_name: str) -> Generator[None, None, None]:
    """Create a test table."""
    try:
        import boto3
    except ImportError:
        raise ImportError(IMPORT_BOTO3_ERROR)

    dynamodb_client = boto3.client(
        "dynamodb",
        endpoint_url="http://localhost:8432",
        region_name="eu-west-2",
        aws_access_key_id="langchain",
        aws_secret_access_key="langchain",
    )

    dynamodb_client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": KEY_FIELD, "KeyType": "HASH"},
            {"AttributeName": NAMESPACE_FIELD, "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": KEY_FIELD, "AttributeType": "S"},
            {"AttributeName": NAMESPACE_FIELD, "AttributeType": "S"},
            {"AttributeName": GROUP_ID_FIELD, "AttributeType": "S"},
            {"AttributeName": UPDATED_AT_FIELD, "AttributeType": "N"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": f"{GROUP_ID_FIELD}-{UPDATED_AT_FIELD}-index",
                "KeySchema": [
                    {"AttributeName": GROUP_ID_FIELD, "KeyType": "HASH"},
                    {"AttributeName": UPDATED_AT_FIELD, "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    waiter = dynamodb_client.get_waiter("table_exists")
    waiter.wait(TableName=table_name)

    yield

    dynamodb_client.delete_table(TableName=table_name)


@pytest.fixture
def record_manager(create_table: None, table_name: str) -> DynamoDBRecordManager:
    """Initialize the test DynamoDB and yield the RecordManager instance."""
    return DynamoDBRecordManager(
        namespace="kittens",
        table_name=table_name,
        endpoint_url="http://localhost:8432",
        aws_access_key_id="langchain",
        aws_secret_access_key="langchain",
        region_name="eu-west-2",
    )


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Vector store fixture."""
    embeddings = DeterministicFakeEmbedding(size=5)
    return InMemoryVectorStore(embeddings)


@pytest.mark.requires("botocore")
def test_update(record_manager: DynamoDBRecordManager) -> None:
    """Test updating records in DynamoDB."""
    read_keys = record_manager.list_keys()
    updated_keys = ["update_key1", "update_key2", "update_key3"]
    record_manager.update(updated_keys)
    all_keys = record_manager.list_keys()
    assert sorted(all_keys) == sorted(read_keys + updated_keys)


@pytest.mark.requires("botocore")
def test_update_timestamp(record_manager: DynamoDBRecordManager) -> None:
    """Test updating records with timestamps in DynamoDB."""
    test_time = datetime(2024, 2, 23).timestamp()

    with patch.object(record_manager, "get_time", return_value=test_time):
        record_manager.update(["key1"])

    response = record_manager.table.get_item(
        Key={"index_key": "key1", "namespace": record_manager.namespace}
    )
    item = response.get("Item", {})

    assert {
        "index_key": item["index_key"],
        "namespace": item["namespace"],
        "updated_at": float(item["updated_at"]),  # type: ignore [arg-type]
        "group_id": item.get("group_id"),
    } == {
        "index_key": "key1",
        "namespace": "kittens",
        "updated_at": test_time,
        "group_id": None,
    }


@pytest.mark.requires("botocore")
def test_exists(record_manager: DynamoDBRecordManager) -> None:
    """Test checking if keys exist in DynamoDB."""
    keys = ["key1", "key2", "key3"]
    record_manager.update(keys)
    exists = record_manager.exists(keys)
    assert len(exists) == len(keys)
    assert all(exists)


@pytest.mark.requires("botocore")
def test_list_keys(record_manager: DynamoDBRecordManager) -> None:
    """Test listing keys in DynamoDB."""
    keys = ["key1", "key2", "key3"]
    record_manager.update(keys)
    listed_keys = record_manager.list_keys()
    assert sorted(listed_keys) == sorted(keys)


@pytest.mark.requires("botocore")
def test_namespace_is_used(record_manager: DynamoDBRecordManager) -> None:
    """Verify that namespace is taken into account for all operations in DynamoDB."""
    # Create a second record_manager with a different namespace
    other_manager = DynamoDBRecordManager(
        namespace="puppies",
        table_name=record_manager.table_name,
        endpoint_url="http://localhost:8432",
        aws_access_key_id="langchain",
        aws_secret_access_key="langchain",
        region_name="us-east-1",
    )

    # Update records in both namespaces
    keys = ["key1", "key2"]
    record_manager.update(keys)
    other_manager.update(keys)

    # Verify that list_keys respects namespaces
    assert sorted(record_manager.list_keys()) == sorted(keys)
    assert sorted(other_manager.list_keys()) == sorted(keys)

    # Verify that exists respects namespaces
    assert all(record_manager.exists(keys))
    assert all(other_manager.exists(keys))

    # Delete in one namespace shouldn't affect the other
    record_manager.delete_keys(keys)
    assert not any(record_manager.exists(keys))
    assert all(other_manager.exists(keys))


@pytest.mark.requires("botocore")
def test_delete_keys(record_manager: DynamoDBRecordManager) -> None:
    """Test deleting keys from DynamoDB."""
    keys = ["key1", "key2"]
    record_manager.update(keys)
    record_manager.delete_keys(keys)
    assert not any(record_manager.exists(keys))


@pytest.mark.requires("botocore")
def test_large_batch_update(record_manager: DynamoDBRecordManager) -> None:
    """Test updating a large batch of records (>25) in DynamoDB."""
    # Create more than MAX_BATCH_SIZE keys
    keys = [f"large_batch_key_{i}" for i in range(30)]
    group_ids = [f"group_{i}" for i in range(30)]

    # Update with group IDs
    record_manager.update(keys, group_ids=group_ids)

    # Verify all keys were written
    exists_results = record_manager.exists(keys)
    assert all(exists_results)

    # Verify group IDs were written correctly
    for key, group_id in zip(keys, group_ids):
        response = record_manager.table.get_item(
            Key={"index_key": key, "namespace": record_manager.namespace}
        )
        item = response.get("Item", {})
        assert item["group_id"] == group_id


@pytest.mark.requires("botocore")
def test_large_batch_exists(record_manager: DynamoDBRecordManager) -> None:
    """Test checking existence of a large batch of records (>25) in DynamoDB."""
    # Create more than MAX_BATCH_SIZE keys
    keys = [f"large_batch_exists_{i}" for i in range(30)]

    # First check non-existent keys
    exists_results = record_manager.exists(keys)
    assert not any(exists_results)

    # Add half the keys
    record_manager.update(keys[:15])

    # Check mixed existence
    exists_results = record_manager.exists(keys)
    assert all(exists_results[:15])  # First 15 should exist
    assert not any(exists_results[15:])  # Rest should not exist


@pytest.mark.requires("botocore")
def test_large_batch_delete(record_manager: DynamoDBRecordManager) -> None:
    """Test deleting a large batch of records (>25) in DynamoDB."""
    # Create more than MAX_BATCH_SIZE keys
    keys = [f"large_batch_delete_{i}" for i in range(30)]
    record_manager.update(keys)

    # Verify all keys exist
    exists_results = record_manager.exists(keys)
    assert all(exists_results)

    # Delete all keys
    record_manager.delete_keys(keys)

    # Verify all keys were deleted
    exists_results = record_manager.exists(keys)
    assert not any(exists_results)


@pytest.mark.requires("botocore")
def test_time_at_least(record_manager: DynamoDBRecordManager) -> None:
    """Test time_at_least parameter behavior."""
    current_time = record_manager.get_time()
    future_time = current_time + 1000

    # Should raise error when time_at_least is in the future
    with pytest.raises(ValueError, match="Server time is behind"):
        record_manager.update(["key1"], time_at_least=future_time)

    # Should succeed when time_at_least is in the past
    past_time = current_time - 1000
    record_manager.update(["key1"], time_at_least=past_time)
    assert record_manager.exists(["key1"])[0]


@pytest.mark.requires("botocore")
def test_list_keys_filters(record_manager: DynamoDBRecordManager) -> None:
    """Test list_keys with various filter combinations."""
    # Setup test data with known timestamps
    with patch.object(record_manager, "get_time") as mock_time:
        # Create records at different times
        mock_time.return_value = 1000.0
        record_manager.update(["old_key1", "old_key2"], group_ids=["group1", "group1"])

        mock_time.return_value = 2000.0
        record_manager.update(["mid_key1", "mid_key2"], group_ids=["group2", "group2"])

        mock_time.return_value = 3000.0
        record_manager.update(["new_key1", "new_key2"], group_ids=["group3", "group3"])

    # Test before filter
    before_keys = record_manager.list_keys(before=2500.0)
    assert sorted(before_keys) == sorted(
        ["old_key1", "old_key2", "mid_key1", "mid_key2"]
    )

    # Test after filter
    after_keys = record_manager.list_keys(after=2500.0)
    assert sorted(after_keys) == sorted(["new_key1", "new_key2"])

    # Test group_ids filter
    group_keys = record_manager.list_keys(group_ids=["group1"])
    assert sorted(group_keys) == sorted(["old_key1", "old_key2"])

    # Test combined filters
    combined_keys = record_manager.list_keys(
        after=1500.0, before=2500.0, group_ids=["group2"]
    )
    assert sorted(combined_keys) == sorted(["mid_key1", "mid_key2"])

    # Test limit
    limited_keys = record_manager.list_keys(limit=2)
    assert len(limited_keys) == 2

    # Test empty results
    empty_keys = record_manager.list_keys(before=0.0)
    assert len(empty_keys) == 0

    empty_group_keys = record_manager.list_keys(group_ids=["nonexistent_group"])
    assert len(empty_group_keys) == 0


@pytest.mark.requires("botocore")
def test_update_group_ids(record_manager: DynamoDBRecordManager) -> None:
    """Test updating group_ids for existing records."""
    # Create initial records
    record_manager.update(["key1", "key2"], group_ids=["group1", "group1"])

    # Update group_id for one key
    record_manager.update(["key1"], group_ids=["group2"])

    # Verify group_id was updated
    response = record_manager.table.get_item(
        Key={"index_key": "key1", "namespace": record_manager.namespace}
    )
    assert response["Item"]["group_id"] == "group2"

    # Verify other key's group_id remained unchanged
    response = record_manager.table.get_item(
        Key={"index_key": "key2", "namespace": record_manager.namespace}
    )
    assert response["Item"]["group_id"] == "group1"


@pytest.mark.requires("botocore")
def test_batch_edge_cases(record_manager: DynamoDBRecordManager) -> None:
    """Test batch operation edge cases."""
    # Test empty batch
    record_manager.update([])
    record_manager.delete_keys([])
    assert record_manager.exists([]) == []

    # Test exactly MAX_BATCH_SIZE items
    keys = [f"key_{i}" for i in range(25)]
    record_manager.update(keys)
    assert all(record_manager.exists(keys))

    # Test duplicate keys in batch
    dup_keys = ["key1", "key1", "key2", "key2"]
    record_manager.update(dup_keys)
    assert record_manager.exists(["key1", "key2"]) == [True, True]


@pytest.mark.requires("botocore")
def test_list_keys_pagination(record_manager: DynamoDBRecordManager) -> None:
    """Test that list_keys properly handles pagination with limits."""
    # Create more than 25 records (DynamoDB's default page size)
    num_records = 30
    keys = [f"key{i}" for i in range(num_records)]

    # Update all records
    record_manager.update(keys=keys)

    # Test with different limit values
    limit_5 = record_manager.list_keys(limit=5)
    assert len(limit_5) == 5, f"Expected 5 keys, got {len(limit_5)}"

    # Get all keys with a limit higher than total records
    limit_50 = record_manager.list_keys(limit=50)
    assert (
        len(limit_50) == num_records
    ), f"Expected {num_records} keys, got {len(limit_50)}"

    # Get exactly the number of records that exist
    limit_30 = record_manager.list_keys(limit=30)
    assert (
        len(limit_30) == num_records
    ), f"Expected {num_records} keys, got {len(limit_30)}"

    # Test with a limit that requires pagination (DynamoDB default page size is 25)
    limit_27 = record_manager.list_keys(limit=27)
    assert len(limit_27) == 27, f"Expected 27 keys, got {len(limit_27)}"

    # Verify that all returned keys are valid
    all_keys = set(keys)
    assert all(key in all_keys for key in limit_5)
    assert all(key in all_keys for key in limit_27)
    assert all(key in all_keys for key in limit_30)
    assert all(key in all_keys for key in limit_50)


class ToyLoader(BaseLoader):
    """Toy loader that always returns the same documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        """Initialize with the documents to return."""
        self.documents = documents

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        yield from self.documents

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:
        for document in self.documents:
            yield document


@pytest.mark.requires("botocore")
def test_indexing_same_content(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Indexing some content to confirm it gets added only once."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
            ),
            Document(
                page_content="This is another document.",
            ),
        ]
    )

    assert index(loader, record_manager, vector_store) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(list(vector_store.store)) == 2

    for _ in range(2):
        # Run the indexing again
        assert index(loader, record_manager, vector_store) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


@pytest.mark.requires("botocore")
def test_index_simple_delete_full(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Indexing some content to confirm it gets added only once."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
            ),
            Document(
                page_content="This is another document.",
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid)["text"]  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"mutated document 1", "This is another document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


@pytest.mark.requires("botocore")
def test_incremental_fails_with_bad_source_ids(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with incremental deletion strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": None},
            ),
        ]
    )

    with pytest.raises(ValueError):
        # Should raise an error because no source id function was specified
        index(loader, record_manager, vector_store, cleanup="incremental")

    with pytest.raises(ValueError):
        # Should raise an error because no source id function was specified
        index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


@pytest.mark.requires("botocore")
def test_no_delete(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing without a deletion strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup=None,
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    # If we add the same content twice it should be skipped
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup=None,
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated content",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
        ]
    )

    # Should result in no updates or deletions!
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup=None,
            source_id_key="source",
        ) == {
            "num_added": 1,
            "num_deleted": 0,
            "num_skipped": 1,
            "num_updated": 0,
        }


@pytest.mark.requires("botocore")
def test_incremental_delete(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with incremental deletion strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid)["text"]  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    # Create 2 documents from the same source all with mutated content
    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="mutated document 2",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "2"},
            ),
        ]
    )

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 3).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid)["text"]  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


@pytest.mark.requires("botocore")
def test_incremental_indexing_with_batch_size(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with incremental indexing"""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="2",
                metadata={"source": "1"},
            ),
            Document(
                page_content="3",
                metadata={"source": "1"},
            ),
            Document(
                page_content="4",
                metadata={"source": "1"},
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=2,
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=2,
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid)["text"]  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"1", "2", "3", "4"}


@pytest.mark.requires("botocore")
def test_incremental_delete_with_batch_size(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with incremental deletion strategy and batch size."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="2",
                metadata={"source": "2"},
            ),
            Document(
                page_content="3",
                metadata={"source": "3"},
            ),
            Document(
                page_content="4",
                metadata={"source": "4"},
            ),
        ]
    )

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=3,
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid)["text"]  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"1", "2", "3", "4"}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=3,
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2022, 1, 3).timestamp()
    ):
        # Docs with same content
        docs = [
            Document(
                page_content="1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="2",
                metadata={"source": "2"},
            ),
        ]
        assert index(
            docs,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=1,
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager, "get_time", return_value=datetime(2023, 1, 3).timestamp()
    ):
        # Docs with same content
        docs = [
            Document(
                page_content="1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="2",
                metadata={"source": "2"},
            ),
        ]
        assert index(
            docs,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=1,
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    # Try to index with changed docs now
    with patch.object(
        record_manager, "get_time", return_value=datetime(2024, 1, 3).timestamp()
    ):
        # Docs with same content
        docs = [
            Document(
                page_content="changed 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="changed 2",
                metadata={"source": "2"},
            ),
        ]
        assert index(
            docs,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 2,
            "num_skipped": 0,
            "num_updated": 0,
        }


@pytest.mark.requires("botocore")
def test_indexing_with_no_docs(
    record_manager: DynamoDBRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert index(loader, record_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("botocore")
def test_deduplication(
    record_manager: DynamoDBRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]

    # Should result in only a single document being added
    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 1,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("botocore")
def test_cleanup_with_different_batchsize(
    record_manager: DynamoDBRecordManager, vector_store: VectorStore
) -> None:
    """Check that we can clean up with different batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": str(d)},
        )
        for d in range(100)
    ]

    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 100,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    docs = [
        Document(
            page_content="Different doc",
            metadata={"source": str(d)},
        )
        for d in range(101)
    ]

    assert index(
        docs, record_manager, vector_store, cleanup="full", cleanup_batch_size=17
    ) == {
        "num_added": 101,
        "num_deleted": 100,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("botocore")
def test_deduplication_v2(
    record_manager: DynamoDBRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="2",
            metadata={"source": "2"},
        ),
        Document(
            page_content="3",
            metadata={"source": "3"},
        ),
    ]

    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 3,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(vector_store, InMemoryVectorStore)
    contents = sorted([document["text"] for document in vector_store.store.values()])
    assert contents == ["1", "2", "3"]


@pytest.mark.requires("botocore")
def test_indexing_force_update(
    record_manager: DynamoDBRecordManager, vector_store: VectorStore
) -> None:
    """Test indexing with force update."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]

    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 2,
        "num_updated": 0,
    }

    assert index(
        docs, record_manager, vector_store, cleanup="full", force_update=True
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 2,
    }


@pytest.mark.requires("botocore")
def test_indexing_custom_batch_size(
    record_manager: DynamoDBRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with a custom batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]
    ids = [_HashedDocument.from_document(doc).uid for doc in docs]

    batch_size = 1
    with patch.object(vector_store, "add_documents") as mock_add_documents:
        index(docs, record_manager, vector_store, batch_size=batch_size)
        args, kwargs = mock_add_documents.call_args
        docs_with_id = [
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
                id=ids[0],
            )
        ]
        assert args == (docs_with_id,)
        assert kwargs == {"ids": ids, "batch_size": batch_size}
