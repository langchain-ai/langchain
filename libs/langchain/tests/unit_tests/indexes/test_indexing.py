from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
)
from unittest.mock import patch

import pytest
import pytest_asyncio

import langchain.vectorstores
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.base import Embeddings
from langchain.indexes import aindex, index
from langchain.indexes._api import _abatch
from langchain.indexes._sql_record_manager import SQLRecordManager
from langchain.schema import Document
from langchain.schema.vectorstore import VST, VectorStore


class ToyLoader(BaseLoader):
    """Toy loader that always returns the same documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        """Initialize with the documents to return."""
        self.documents = documents

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        yield from self.documents

    def load(self) -> List[Document]:
        """Load the documents from the source."""
        return list(self.lazy_load())

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:
        async def async_generator() -> AsyncIterator[Document]:
            for document in self.documents:
                yield document

        return async_generator()

    async def aload(self) -> List[Document]:
        """Load the documents from the source."""
        return [doc async for doc in await self.alazy_load()]


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore using a dictionary."""

    def __init__(self) -> None:
        """Vector store interface for testing things in memory."""
        self.store: Dict[str, Document] = {}

    def delete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    async def adelete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    def add_documents(  # type: ignore
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Add the given documents to the store (insert behavior)."""
        if ids and len(ids) != len(documents):
            raise ValueError(
                f"Expected {len(ids)} ids, got {len(documents)} documents."
            )

        if not ids:
            raise NotImplementedError("This is not implemented yet.")

        for _id, document in zip(ids, documents):
            if _id in self.store:
                raise ValueError(
                    f"Document with uid {_id} already exists in the store."
                )
            self.store[_id] = document

    async def aadd_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids and len(ids) != len(documents):
            raise ValueError(
                f"Expected {len(ids)} ids, got {len(documents)} documents."
            )

        if not ids:
            raise NotImplementedError("This is not implemented yet.")

        for _id, document in zip(ids, documents):
            if _id in self.store:
                raise ValueError(
                    f"Document with uid {_id} already exists in the store."
                )
            self.store[_id] = document
        return list(ids)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given texts to the store (insert behavior)."""
        raise NotImplementedError()

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> VST:
        """Create a vector store from a list of texts."""
        raise NotImplementedError()

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Find the most similar documents to the given query."""
        raise NotImplementedError()


@pytest.fixture
def record_manager() -> SQLRecordManager:
    """Timestamped set fixture."""
    record_manager = SQLRecordManager("kittens", db_url="sqlite:///:memory:")
    record_manager.create_schema()
    return record_manager


@pytest_asyncio.fixture  # type: ignore
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


def test_indexing_same_content(
    record_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_aindexing_same_content(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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

    assert await aindex(await loader.alazy_load(), arecord_manager, vector_store) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(list(vector_store.store)) == 2

    for _ in range(2):
        # Run the indexing again
        assert await aindex(
            await loader.alazy_load(), arecord_manager, vector_store
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_index_simple_delete_full(
    record_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        vector_store.store.get(uid).page_content  # type: ignore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_aindex_simple_delete_full(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(), arecord_manager, vector_store, cleanup="full"
        ) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(), arecord_manager, vector_store, cleanup="full"
        ) == {
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(), arecord_manager, vector_store, cleanup="full"
        ) == {
            "num_added": 1,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"mutated document 1", "This is another document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(), arecord_manager, vector_store, cleanup="full"
        ) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_incremental_fails_with_bad_source_ids(
    record_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_aincremental_fails_with_bad_source_ids(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        await aindex(
            await loader.alazy_load(),
            arecord_manager,
            vector_store,
            cleanup="incremental",
        )

    with pytest.raises(ValueError):
        # Should raise an error because no source id function was specified
        await aindex(
            await loader.alazy_load(),
            arecord_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
        )


def test_no_delete(
    record_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_ano_delete(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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
    record_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        vector_store.store.get(uid).page_content  # type: ignore
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
        vector_store.store.get(uid).page_content  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_aincremental_delete(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 2).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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
        arecord_manager, "aget_time", return_value=datetime(2021, 1, 3).timestamp()
    ):
        assert await aindex(
            await loader.alazy_load(),
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

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        vector_store.store.get(uid).page_content  # type: ignore
        for uid in vector_store.store
    )
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_indexing_with_no_docs(
    record_manager: SQLRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert index(loader, record_manager, vector_store, cleanup="full") == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_aindexing_with_no_docs(
    arecord_manager: SQLRecordManager, vector_store: VectorStore
) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert await aindex(
        await loader.alazy_load(), arecord_manager, vector_store, cleanup="full"
    ) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication(
    record_manager: SQLRecordManager, vector_store: VectorStore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_adeduplication(
    arecord_manager: SQLRecordManager, vector_store: VectorStore
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
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_cleanup_with_different_batchsize(
    record_manager: SQLRecordManager, vector_store: VectorStore
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


@pytest.mark.asyncio
@pytest.mark.requires("aiosqlite")
async def test_async_cleanup_with_different_batchsize(
    arecord_manager: SQLRecordManager, vector_store: InMemoryVectorStore
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


def test_deduplication_v2(
    record_manager: SQLRecordManager, vector_store: VectorStore
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
    contents = sorted(
        [document.page_content for document in vector_store.store.values()]
    )
    assert contents == ["1", "2", "3"]


async def _to_async_iter(it: Iterable[Any]) -> AsyncIterator[Any]:
    """Convert an iterable to an async iterator."""
    for i in it:
        yield i


@pytest.mark.asyncio
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


def test_compatible_vectorstore_documentation() -> None:
    """Test which vectorstores are compatible with the indexing API.

    This serves as a reminder to update the documentation in [1]
    that specifies which vectorstores are compatible with the
    indexing API.

    Ideally if a developer adds a new vectorstore or modifies
    an existing one in such a way that affects its compatibility
    with the Indexing API, he/she will see this failed test
    case and 1) update docs in [1] and 2) update the `documented`
    dict in this test case.

    [1] langchain/docs/docs_skeleton/docs/modules/data_connection/indexing.ipynb
    """

    # Check if a vectorstore is compatible with the indexing API
    def check_compatibility(vector_store: VectorStore) -> bool:
        """Check if a vectorstore is compatible with the indexing API."""
        methods = ["delete", "add_documents"]
        for method in methods:
            if not hasattr(vector_store, method):
                return False
        # Checking if the vectorstore has overridden the default delete method
        # implementation which just raises a NotImplementedError
        if getattr(vector_store, "delete") == VectorStore.delete:
            return False
        return True

    # Check all vector store classes for compatibility
    compatible = set()
    for class_name in langchain.vectorstores.__all__:
        # Get the definition of the class
        cls = getattr(langchain.vectorstores, class_name)

        # If the class corresponds to a vectorstore, check its compatibility
        if issubclass(cls, VectorStore):
            is_compatible = check_compatibility(cls)
            if is_compatible:
                compatible.add(class_name)

    # These are mentioned in the indexing.ipynb documentation
    documented = {
        "AnalyticDB",
        "AzureCosmosDBVectorSearch",
        "AwaDB",
        "Bagel",
        "Cassandra",
        "Chroma",
        "DashVector",
        "DeepLake",
        "Dingo",
        "ElasticVectorSearch",
        "ElasticsearchStore",
        "FAISS",
        "MomentoVectorIndex",
        "PGVector",
        "Pinecone",
        "Qdrant",
        "Redis",
        "ScaNN",
        "SemaDB",
        "SupabaseVectorStore",
        "TimescaleVector",
        "Vald",
        "Vearch",
        "VespaStore",
        "Weaviate",
        "ZepVectorStore",
    }
    assert compatible == documented
