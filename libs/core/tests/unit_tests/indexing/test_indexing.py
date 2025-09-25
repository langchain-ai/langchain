from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
)
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.indexing import InMemoryRecordManager, aindex, index
from langchain_core.indexing.api import (
    IndexingException,
    _abatch,
    _get_document_with_hash,
)
from langchain_core.indexing.in_memory import InMemoryDocumentIndex
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


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


@pytest.fixture
def record_manager() -> InMemoryRecordManager:
    """Timestamped set fixture."""
    record_manager = InMemoryRecordManager(namespace="hello")
    record_manager.create_schema()
    return record_manager


@pytest_asyncio.fixture
async def arecord_manager() -> InMemoryRecordManager:
    """Timestamped set fixture."""
    record_manager = InMemoryRecordManager(namespace="hello")
    await record_manager.acreate_schema()
    return record_manager


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Vector store fixture."""
    embeddings = DeterministicFakeEmbedding(size=5)
    return InMemoryVectorStore(embeddings)


@pytest.fixture
def upserting_vector_store() -> InMemoryVectorStore:
    """Vector store fixture."""
    embeddings = DeterministicFakeEmbedding(size=5)
    return InMemoryVectorStore(embeddings)


def test_indexing_same_content(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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


async def test_aindexing_same_content(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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

    assert await aindex(loader, arecord_manager, vector_store) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(list(vector_store.store)) == 2

    for _ in range(2):
        # Run the indexing again
        assert await aindex(loader, arecord_manager, vector_store) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_index_simple_delete_full(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        indexing_result = index(loader, record_manager, vector_store, cleanup="full")

        doc_texts = {
            # Ignoring type since doc should be in the store and not a None
            vector_store.get_by_ids([uid])[0].page_content
            for uid in vector_store.store
        }
        assert doc_texts == {"mutated document 1", "This is another document."}

        assert indexing_result == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


async def test_aindex_simple_delete_full(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"mutated document 1", "This is another document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_index_delete_full_recovery_after_deletion_failure(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
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

    with (
        patch.object(
            record_manager,
            "get_time",
            return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
        ),
        patch.object(vector_store, "delete", return_value=False),
        pytest.raises(IndexingException),
    ):
        indexing_result = index(loader, record_manager, vector_store, cleanup="full")

    # At this point, there should be 3 records in both the record manager
    # and the vector store
    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {
        "This is a test document.",
        "mutated document 1",
        "This is another document.",
    }

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        indexing_result = index(loader, record_manager, vector_store, cleanup="full")
        doc_texts = {
            # Ignoring type since doc should be in the store and not a None
            vector_store.get_by_ids([uid])[0].page_content
            for uid in vector_store.store
        }
        assert doc_texts == {"mutated document 1", "This is another document."}

    assert indexing_result == {
        "num_added": 0,
        "num_deleted": 1,
        "num_skipped": 2,
        "num_updated": 0,
    }


async def test_aindex_delete_full_recovery_after_deletion_failure(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
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

    with (
        patch.object(
            arecord_manager,
            "get_time",
            return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
        ),
        patch.object(vector_store, "adelete", return_value=False),
        pytest.raises(IndexingException),
    ):
        indexing_result = await aindex(
            loader, arecord_manager, vector_store, cleanup="full"
        )

    # At this point, there should be 3 records in both the record manager
    # and the vector store
    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {
        "This is a test document.",
        "mutated document 1",
        "This is another document.",
    }

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        indexing_result = await aindex(
            loader, arecord_manager, vector_store, cleanup="full"
        )
        doc_texts = {
            # Ignoring type since doc should be in the store and not a None
            vector_store.get_by_ids([uid])[0].page_content
            for uid in vector_store.store
        }
        assert doc_texts == {"mutated document 1", "This is another document."}

    assert indexing_result == {
        "num_added": 0,
        "num_deleted": 1,
        "num_skipped": 2,
        "num_updated": 0,
    }


def test_incremental_fails_with_bad_source_ids(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode is "
        "incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(loader, record_manager, vector_store, cleanup="incremental")

    with pytest.raises(
        ValueError,
        match="Source ids are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


async def test_aincremental_fails_with_bad_source_ids(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode "
        "is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="incremental",
        )

    with pytest.raises(
        ValueError,
        match="Source ids are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


def test_index_simple_delete_scoped_full(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is a test document from another source.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "1"},
            ),
        ]
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 1,
            "num_deleted": 2,
            "num_skipped": 1,
            "num_updated": 0,
        }
        doc_texts = {
            # Ignoring type since doc should be in the store and not a None
            vector_store.get_by_ids([uid])[0].page_content
            for uid in vector_store.store
        }
        assert doc_texts == {
            "mutated document 1",
            "This is another document.",
            "This is a test document from another source.",
        }

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


async def test_aindex_simple_delete_scoped_full(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is a test document from another source.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "1"},
            ),
        ]
    )

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 1,
            "num_deleted": 2,
            "num_skipped": 1,
            "num_updated": 0,
        }
        doc_texts = {
            # Ignoring type since doc should be in the store and not a None
            vector_store.get_by_ids([uid])[0].page_content
            for uid in vector_store.store
        }
        assert doc_texts == {
            "mutated document 1",
            "This is another document.",
            "This is a test document from another source.",
        }

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_scoped_full_fails_with_bad_source_ids(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
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

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode "
        "is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(loader, record_manager, vector_store, cleanup="scoped_full")

    with pytest.raises(
        ValueError,
        match="Source ids are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        )


async def test_ascoped_full_fails_with_bad_source_ids(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
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

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode "
        "is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        await aindex(loader, arecord_manager, vector_store, cleanup="scoped_full")

    with pytest.raises(
        ValueError,
        match="Source ids are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        )


def test_index_empty_doc_scoped_full(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is a test document from another source.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    loader = ToyLoader(documents=[])

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }


async def test_aindex_empty_doc_scoped_full(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test Indexing with scoped_full strategy."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is a test document from another source.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 4,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 4,
            "num_updated": 0,
        }

    loader = ToyLoader(documents=[])

    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="scoped_full",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }


def test_no_delete(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
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


async def test_ano_delete(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup=None,
            source_id_key="source",
        ) == {
            "num_added": 1,
            "num_deleted": 0,
            "num_skipped": 1,
            "num_updated": 0,
        }


def test_incremental_delete(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_incremental_delete_with_same_source(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
                metadata={"source": "1"},
            ),
        ]
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Delete 1 document and unchange 1 document
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "1"},
            ),
        ]
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 0,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {
        "This is another document.",
    }


def test_incremental_indexing_with_batch_size(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with incremental indexing."""
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=2,
        ) == {
            "num_added": 2,
            "num_deleted": 2,
            "num_skipped": 2,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}


def test_incremental_delete_with_batch_size(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2022, 1, 3, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2023, 1, 4, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    # Try to index with changed docs now
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp(),
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"changed 1", "changed 2", "3", "4"}


async def test_aincremental_delete(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader.lazy_load(),
            arecord_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader.lazy_load(),
            arecord_manager,
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
        arecord_manager,
        "get_time",
        return_value=datetime(2021, 1, 3, tzinfo=timezone.utc).timestamp(),
    ):
        assert await aindex(
            loader.lazy_load(),
            arecord_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        ) == {
            "num_added": 2,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.get_by_ids([uid])[0].page_content
        for uid in vector_store.store
    }
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_indexing_with_no_docs(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert index(loader, record_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


async def test_aindexing_with_no_docs(
    arecord_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
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
        "num_skipped": 1,
        "num_updated": 0,
    }


async def test_adeduplication(
    arecord_manager: InMemoryRecordManager, vector_store: VectorStore
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
    assert await aindex(docs, arecord_manager, vector_store, cleanup="full") == {
        "num_added": 1,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 0,
    }


def test_within_batch_deduplication_counting(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Test that within-batch deduplicated documents are counted in num_skipped."""
    # Create documents with within-batch duplicates
    docs = [
        Document(
            page_content="Document A",
            metadata={"source": "1"},
        ),
        Document(
            page_content="Document A",  # Duplicate in same batch
            metadata={"source": "1"},
        ),
        Document(
            page_content="Document B",
            metadata={"source": "2"},
        ),
        Document(
            page_content="Document B",  # Duplicate in same batch
            metadata={"source": "2"},
        ),
        Document(
            page_content="Document C",
            metadata={"source": "3"},
        ),
    ]

    # Index with large batch size to ensure all docs are in one batch
    result = index(
        docs,
        record_manager,
        vector_store,
        batch_size=10,  # All docs in one batch
        cleanup="full",
    )

    # Should have 3 unique documents added
    assert result["num_added"] == 3
    # Should have 2 documents skipped due to within-batch deduplication
    assert result["num_skipped"] == 2
    # Total should match input
    assert result["num_added"] + result["num_skipped"] == len(docs)
    assert result["num_deleted"] == 0
    assert result["num_updated"] == 0

    # Verify the content
    assert isinstance(vector_store, InMemoryVectorStore)
    ids = list(vector_store.store.keys())
    contents = sorted(
        [document.page_content for document in vector_store.get_by_ids(ids)]
    )
    assert contents == ["Document A", "Document B", "Document C"]


async def test_awithin_batch_deduplication_counting(
    arecord_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Test that within-batch deduplicated documents are counted in num_skipped."""
    # Create documents with within-batch duplicates
    docs = [
        Document(
            page_content="Document A",
            metadata={"source": "1"},
        ),
        Document(
            page_content="Document A",  # Duplicate in same batch
            metadata={"source": "1"},
        ),
        Document(
            page_content="Document B",
            metadata={"source": "2"},
        ),
        Document(
            page_content="Document B",  # Duplicate in same batch
            metadata={"source": "2"},
        ),
        Document(
            page_content="Document C",
            metadata={"source": "3"},
        ),
    ]

    # Index with large batch size to ensure all docs are in one batch
    result = await aindex(
        docs,
        arecord_manager,
        vector_store,
        batch_size=10,  # All docs in one batch
        cleanup="full",
    )

    # Should have 3 unique documents added
    assert result["num_added"] == 3
    # Should have 2 documents skipped due to within-batch deduplication
    assert result["num_skipped"] == 2
    # Total should match input
    assert result["num_added"] + result["num_skipped"] == len(docs)
    assert result["num_deleted"] == 0
    assert result["num_updated"] == 0

    # Verify the content
    assert isinstance(vector_store, InMemoryVectorStore)
    ids = list(vector_store.store.keys())
    contents = sorted(
        [document.page_content for document in vector_store.get_by_ids(ids)]
    )
    assert contents == ["Document A", "Document B", "Document C"]


def test_full_cleanup_with_different_batchsize(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Check that we can clean up with different batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": str(d)},
        )
        for d in range(1000)
    ]

    assert index(docs, record_manager, vector_store, cleanup="full") == {
        "num_added": 1000,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    docs = [
        Document(
            page_content="Different doc",
            metadata={"source": str(d)},
        )
        for d in range(1001)
    ]

    assert index(
        docs, record_manager, vector_store, cleanup="full", cleanup_batch_size=17
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_incremental_cleanup_with_different_batchsize(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
) -> None:
    """Check that we can clean up with different batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": str(d)},
        )
        for d in range(1000)
    ]

    assert index(
        docs,
        record_manager,
        vector_store,
        source_id_key="source",
        cleanup="incremental",
    ) == {
        "num_added": 1000,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    docs = [
        Document(
            page_content="Different doc",
            metadata={"source": str(d)},
        )
        for d in range(1001)
    ]

    assert index(
        docs,
        record_manager,
        vector_store,
        source_id_key="source",
        cleanup="incremental",
        cleanup_batch_size=17,
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


async def test_afull_cleanup_with_different_batchsize(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Check that we can clean up with different batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": str(d)},
        )
        for d in range(1000)
    ]

    assert await aindex(docs, arecord_manager, vector_store, cleanup="full") == {
        "num_added": 1000,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    docs = [
        Document(
            page_content="Different doc",
            metadata={"source": str(d)},
        )
        for d in range(1001)
    ]

    assert await aindex(
        docs, arecord_manager, vector_store, cleanup="full", cleanup_batch_size=17
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


async def test_aincremental_cleanup_with_different_batchsize(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Check that we can clean up with different batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": str(d)},
        )
        for d in range(1000)
    ]

    assert await aindex(
        docs,
        arecord_manager,
        vector_store,
        source_id_key="source",
        cleanup="incremental",
    ) == {
        "num_added": 1000,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    docs = [
        Document(
            page_content="Different doc",
            metadata={"source": str(d)},
        )
        for d in range(1001)
    ]

    assert await aindex(
        docs,
        arecord_manager,
        vector_store,
        cleanup="incremental",
        source_id_key="source",
        cleanup_batch_size=17,
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication_v2(
    record_manager: InMemoryRecordManager, vector_store: VectorStore
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
        "num_skipped": 1,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(vector_store, InMemoryVectorStore)

    ids = list(vector_store.store.keys())
    contents = sorted(
        [document.page_content for document in vector_store.get_by_ids(ids)]
    )
    assert contents == ["1", "2", "3"]


async def _to_async_iter(it: Iterable[Any]) -> AsyncIterator[Any]:
    """Convert an iterable to an async iterator."""
    for i in it:
        yield i


async def test_abatch() -> None:
    """Test the abatch function."""
    batches = _abatch(5, _to_async_iter(range(12)))
    assert isinstance(batches, AsyncIterator)
    assert [batch async for batch in batches] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11],
    ]

    batches = _abatch(1, _to_async_iter(range(3)))
    assert isinstance(batches, AsyncIterator)
    assert [batch async for batch in batches] == [[0], [1], [2]]

    batches = _abatch(2, _to_async_iter(range(5)))
    assert isinstance(batches, AsyncIterator)
    assert [batch async for batch in batches] == [[0, 1], [2, 3], [4]]


def test_indexing_force_update(
    record_manager: InMemoryRecordManager, upserting_vector_store: VectorStore
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

    assert index(docs, record_manager, upserting_vector_store, cleanup="full") == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 0,
    }

    assert index(docs, record_manager, upserting_vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 3,
        "num_updated": 0,
    }

    assert index(
        docs, record_manager, upserting_vector_store, cleanup="full", force_update=True
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 2,
    }


async def test_aindexing_force_update(
    arecord_manager: InMemoryRecordManager, upserting_vector_store: VectorStore
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

    assert await aindex(
        docs, arecord_manager, upserting_vector_store, cleanup="full"
    ) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 0,
    }

    assert await aindex(
        docs, arecord_manager, upserting_vector_store, cleanup="full"
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 3,
        "num_updated": 0,
    }

    assert await aindex(
        docs,
        arecord_manager,
        upserting_vector_store,
        cleanup="full",
        force_update=True,
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 2,
    }


def test_indexing_custom_batch_size(
    record_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with a custom batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]
    ids = [_get_document_with_hash(doc, key_encoder="sha256").id for doc in docs]

    batch_size = 1

    original = vector_store.add_documents

    try:
        mock_add_documents = MagicMock()
        vector_store.add_documents = mock_add_documents  # type: ignore[method-assign]

        index(
            docs,
            record_manager,
            vector_store,
            batch_size=batch_size,
            key_encoder="sha256",
        )
        args, kwargs = mock_add_documents.call_args
        doc_with_id = Document(
            id=ids[0], page_content="This is a test document.", metadata={"source": "1"}
        )
        assert args == ([doc_with_id],)
        assert kwargs == {"ids": ids, "batch_size": batch_size}
    finally:
        vector_store.add_documents = original  # type: ignore[method-assign]


async def test_aindexing_custom_batch_size(
    arecord_manager: InMemoryRecordManager, vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with a custom batch size."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]
    ids = [_get_document_with_hash(doc, key_encoder="sha256").id for doc in docs]

    batch_size = 1
    mock_add_documents = AsyncMock()
    doc_with_id = Document(
        id=ids[0], page_content="This is a test document.", metadata={"source": "1"}
    )
    vector_store.aadd_documents = mock_add_documents  # type: ignore[method-assign]
    await aindex(
        docs, arecord_manager, vector_store, batch_size=batch_size, key_encoder="sha256"
    )
    args, kwargs = mock_add_documents.call_args
    assert args == ([doc_with_id],)
    assert kwargs == {"ids": ids, "batch_size": batch_size}


def test_index_into_document_index(record_manager: InMemoryRecordManager) -> None:
    """Get an in memory index."""
    document_index = InMemoryDocumentIndex()
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
    ]

    assert index(docs, record_manager, document_index, cleanup="full") == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert index(docs, record_manager, document_index, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 2,
        "num_updated": 0,
    }

    assert index(
        docs, record_manager, document_index, cleanup="full", force_update=True
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 2,
    }

    assert index([], record_manager, document_index, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 2,
        "num_skipped": 0,
        "num_updated": 0,
    }


async def test_aindex_into_document_index(
    arecord_manager: InMemoryRecordManager,
) -> None:
    """Get an in memory index."""
    document_index = InMemoryDocumentIndex()
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
    ]

    assert await aindex(
        docs,
        arecord_manager,
        document_index,
        cleanup="full",
        key_encoder="sha256",
    ) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert await aindex(
        docs,
        arecord_manager,
        document_index,
        cleanup="full",
        key_encoder="sha256",
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 2,
        "num_updated": 0,
    }

    assert await aindex(
        docs,
        arecord_manager,
        document_index,
        cleanup="full",
        force_update=True,
        key_encoder="sha256",
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 2,
    }

    assert await aindex(
        [],
        arecord_manager,
        document_index,
        cleanup="full",
        key_encoder="sha256",
    ) == {
        "num_added": 0,
        "num_deleted": 2,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_index_with_upsert_kwargs(
    record_manager: InMemoryRecordManager, upserting_vector_store: InMemoryVectorStore
) -> None:
    """Test indexing with upsert_kwargs parameter."""
    mock_add_documents = MagicMock()

    with patch.object(upserting_vector_store, "add_documents", mock_add_documents):
        docs = [
            Document(
                page_content="Test document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="Test document 2",
                metadata={"source": "2"},
            ),
        ]

        upsert_kwargs = {"vector_field": "embedding"}

        index(docs, record_manager, upserting_vector_store, upsert_kwargs=upsert_kwargs)

        # Assert that add_documents was called with the correct arguments
        mock_add_documents.assert_called_once()
        call_args = mock_add_documents.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check that the documents are correct (ignoring ids)
        assert len(args[0]) == 2
        assert all(isinstance(doc, Document) for doc in args[0])
        assert [doc.page_content for doc in args[0]] == [
            "Test document 1",
            "Test document 2",
        ]
        assert [doc.metadata for doc in args[0]] == [{"source": "1"}, {"source": "2"}]

        # Check that ids are present
        assert "ids" in kwargs
        assert isinstance(kwargs["ids"], list)
        assert len(kwargs["ids"]) == 2

        # Check other arguments
        assert kwargs["batch_size"] == 100
        assert kwargs["vector_field"] == "embedding"


def test_index_with_upsert_kwargs_for_document_indexer(
    record_manager: InMemoryRecordManager,
    mocker: MockerFixture,
) -> None:
    """Test that kwargs are passed to the upsert method of the document indexer."""
    document_index = InMemoryDocumentIndex()
    upsert_spy = mocker.spy(document_index.__class__, "upsert")
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
    ]

    upsert_kwargs = {"vector_field": "embedding"}

    assert index(
        docs,
        record_manager,
        document_index,
        cleanup="full",
        upsert_kwargs=upsert_kwargs,
    ) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert upsert_spy.call_count == 1
    # assert call kwargs were passed as kwargs
    assert upsert_spy.call_args.kwargs == upsert_kwargs


async def test_aindex_with_upsert_kwargs_for_document_indexer(
    arecord_manager: InMemoryRecordManager,
    mocker: MockerFixture,
) -> None:
    """Test that kwargs are passed to the upsert method of the document indexer."""
    document_index = InMemoryDocumentIndex()
    upsert_spy = mocker.spy(document_index.__class__, "aupsert")
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is another document.",
            metadata={"source": "2"},
        ),
    ]

    upsert_kwargs = {"vector_field": "embedding"}

    assert await aindex(
        docs,
        arecord_manager,
        document_index,
        cleanup="full",
        upsert_kwargs=upsert_kwargs,
    ) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert upsert_spy.call_count == 1
    # assert call kwargs were passed as kwargs
    assert upsert_spy.call_args.kwargs == upsert_kwargs


async def test_aindex_with_upsert_kwargs(
    arecord_manager: InMemoryRecordManager, upserting_vector_store: InMemoryVectorStore
) -> None:
    """Test async indexing with upsert_kwargs parameter."""
    mock_aadd_documents = AsyncMock()

    with patch.object(upserting_vector_store, "aadd_documents", mock_aadd_documents):
        docs = [
            Document(
                page_content="Async test document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="Async test document 2",
                metadata={"source": "2"},
            ),
        ]

        upsert_kwargs = {"vector_field": "embedding"}

        await aindex(
            docs,
            arecord_manager,
            upserting_vector_store,
            upsert_kwargs=upsert_kwargs,
        )

        # Assert that aadd_documents was called with the correct arguments
        mock_aadd_documents.assert_called_once()
        call_args = mock_aadd_documents.call_args
        assert call_args is not None
        args, kwargs = call_args

        # Check that the documents are correct (ignoring ids)
        assert len(args[0]) == 2
        assert all(isinstance(doc, Document) for doc in args[0])
        assert [doc.page_content for doc in args[0]] == [
            "Async test document 1",
            "Async test document 2",
        ]
        assert [doc.metadata for doc in args[0]] == [{"source": "1"}, {"source": "2"}]

        # Check that ids are present
        assert "ids" in kwargs
        assert isinstance(kwargs["ids"], list)
        assert len(kwargs["ids"]) == 2

        # Check other arguments
        assert kwargs["batch_size"] == 100
        assert kwargs["vector_field"] == "embedding"
