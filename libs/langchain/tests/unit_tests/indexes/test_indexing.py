from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
)
from unittest.mock import patch

import pytest
import pytest_asyncio
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing.api import _abatch, _get_document_with_hash
from langchain_core.vectorstores import VST, VectorStore
from typing_extensions import override

from langchain_classic.indexes import aindex, index
from langchain_classic.indexes._sql_record_manager import SQLRecordManager


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


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore using a dictionary."""

    def __init__(self, *, permit_upserts: bool = False) -> None:
        """Vector store interface for testing things in memory."""
        self.store: dict[str, Document] = {}
        self.permit_upserts = permit_upserts

    @override
    def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    @override
    async def adelete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    @override
    def add_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add the given documents to the store (insert behavior)."""
        if ids and len(ids) != len(documents):
            msg = f"Expected {len(ids)} ids, got {len(documents)} documents."
            raise ValueError(msg)

        if not ids:
            msg = "This is not implemented yet."
            raise NotImplementedError(msg)

        for _id, document in zip(ids, documents, strict=False):
            if _id in self.store and not self.permit_upserts:
                msg = f"Document with uid {_id} already exists in the store."
                raise ValueError(msg)
            self.store[_id] = document

        return list(ids)

    @override
    async def aadd_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if ids and len(ids) != len(documents):
            msg = f"Expected {len(ids)} ids, got {len(documents)} documents."
            raise ValueError(msg)

        if not ids:
            msg = "This is not implemented yet."
            raise NotImplementedError(msg)

        for _id, document in zip(ids, documents, strict=False):
            if _id in self.store and not self.permit_upserts:
                msg = f"Document with uid {_id} already exists in the store."
                raise ValueError(msg)
            self.store[_id] = document
        return list(ids)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add the given texts to the store (insert behavior)."""
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> VST:
        """Create a vector store from a list of texts."""
        raise NotImplementedError

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Find the most similar documents to the given query."""
        raise NotImplementedError


@pytest.fixture
def record_manager() -> SQLRecordManager:
    """Timestamped set fixture."""
    record_manager = SQLRecordManager("kittens", db_url="sqlite:///:memory:")
    record_manager.create_schema()
    return record_manager


@pytest_asyncio.fixture
@pytest.mark.requires("aiosqlite")
async def arecord_manager() -> SQLRecordManager:
    """Timestamped set fixture."""
    record_manager = SQLRecordManager(
        "kittens",
        db_url="sqlite+aiosqlite:///:memory:",
        async_mode=True,
    )
    await record_manager.acreate_schema()
    return record_manager


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Vector store fixture."""
    return InMemoryVectorStore()


@pytest.fixture
def upserting_vector_store() -> InMemoryVectorStore:
    """Vector store fixture."""
    return InMemoryVectorStore(permit_upserts=True)


_JANUARY_FIRST = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
_JANUARY_SECOND = datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp()


def test_indexing_same_content(
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
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


@pytest.mark.requires("aiosqlite")
async def test_aindexing_same_content(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
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
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_FIRST,
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
        return_value=_JANUARY_FIRST,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"mutated document 1", "This is another document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
    ):
        assert index(loader, record_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


@pytest.mark.requires("aiosqlite")
async def test_aindex_simple_delete_full(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_FIRST,
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_FIRST,
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
        ],
    )

    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"mutated document 1", "This is another document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
    ):
        assert await aindex(loader, arecord_manager, vector_store, cleanup="full") == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_incremental_fails_with_bad_source_ids(
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode is incremental "
        "or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(loader, record_manager, vector_store, cleanup="incremental")

    with pytest.raises(
        ValueError,
        match="Source IDs are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        index(
            loader,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


@pytest.mark.requires("aiosqlite")
async def test_aincremental_fails_with_bad_source_ids(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with pytest.raises(
        ValueError,
        match="Source id key is required when cleanup mode is incremental "
        "or scoped_full",
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
        match="Source IDs are required when cleanup mode is incremental or scoped_full",
    ):
        # Should raise an error because no source id function was specified
        await aindex(
            loader,
            arecord_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


def test_no_delete(
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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
        return_value=_JANUARY_SECOND,
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
        ],
    )

    # Should result in no updates or deletions!
    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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


@pytest.mark.requires("aiosqlite")
async def test_ano_delete(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
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
        "aget_time",
        return_value=_JANUARY_SECOND,
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
        ],
    )

    # Should result in no updates or deletions!
    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
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
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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
        ],
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
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_incremental_indexing_with_batch_size(
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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

    doc_texts = {
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}


def test_incremental_delete_with_batch_size(
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"1", "2", "3", "4"}

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=_JANUARY_SECOND,
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

    # Attempt to index again verify that nothing changes
    with patch.object(
        record_manager,
        "get_time",
        return_value=datetime(2023, 1, 3, tzinfo=timezone.utc).timestamp(),
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
        record_manager,
        "get_time",
        return_value=datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp(),
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


@pytest.mark.requires("aiosqlite")
async def test_aincremental_delete(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        ],
    )

    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
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
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "aget_time",
        return_value=_JANUARY_SECOND,
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
        ],
    )

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager,
        "aget_time",
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
        vector_store.store.get(uid).page_content  # type: ignore[union-attr]
        for uid in vector_store.store
    }
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_indexing_with_no_docs(
    record_manager: SQLRecordManager,
    vector_store: VectorStore,
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert index(loader, record_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
async def test_aindexing_with_no_docs(
    arecord_manager: SQLRecordManager,
    vector_store: VectorStore,
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
    record_manager: SQLRecordManager,
    vector_store: VectorStore,
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


@pytest.mark.requires("aiosqlite")
async def test_adeduplication(
    arecord_manager: SQLRecordManager,
    vector_store: VectorStore,
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


def test_cleanup_with_different_batchsize(
    record_manager: SQLRecordManager,
    vector_store: VectorStore,
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
        docs,
        record_manager,
        vector_store,
        cleanup="full",
        cleanup_batch_size=17,
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
async def test_async_cleanup_with_different_batchsize(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
        docs,
        arecord_manager,
        vector_store,
        cleanup="full",
        cleanup_batch_size=17,
    ) == {
        "num_added": 1001,
        "num_deleted": 1000,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication_v2(
    record_manager: SQLRecordManager,
    vector_store: VectorStore,
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
    contents = sorted(
        [document.page_content for document in vector_store.store.values()],
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
    record_manager: SQLRecordManager,
    upserting_vector_store: VectorStore,
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
        docs,
        record_manager,
        upserting_vector_store,
        cleanup="full",
        force_update=True,
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 2,
    }


@pytest.mark.requires("aiosqlite")
async def test_aindexing_force_update(
    arecord_manager: SQLRecordManager,
    upserting_vector_store: VectorStore,
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
        docs,
        arecord_manager,
        upserting_vector_store,
        cleanup="full",
    ) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 1,
        "num_updated": 0,
    }

    assert await aindex(
        docs,
        arecord_manager,
        upserting_vector_store,
        cleanup="full",
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
    record_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
    with patch.object(vector_store, "add_documents") as mock_add_documents:
        index(
            docs,
            record_manager,
            vector_store,
            batch_size=batch_size,
            key_encoder="sha256",
        )
        args, kwargs = mock_add_documents.call_args
        docs_with_id = [
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
                id=ids[0],
            ),
        ]
        assert args == (docs_with_id,)
        assert kwargs == {"ids": ids, "batch_size": batch_size}


@pytest.mark.requires("aiosqlite")
async def test_aindexing_custom_batch_size(
    arecord_manager: SQLRecordManager,
    vector_store: InMemoryVectorStore,
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
    with patch.object(vector_store, "aadd_documents") as mock_add_documents:
        await aindex(
            docs,
            arecord_manager,
            vector_store,
            batch_size=batch_size,
            key_encoder="sha256",
        )
        args, kwargs = mock_add_documents.call_args
        docs_with_id = [
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
                id=ids[0],
            ),
        ]
        assert args == (docs_with_id,)
        assert kwargs == {"ids": ids, "batch_size": batch_size}
