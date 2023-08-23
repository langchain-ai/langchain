from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Type
from unittest.mock import patch

import pytest

from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.base import Embeddings
from langchain.indexes import index
from langchain.indexes._sql_record_manager import SQLRecordManager
from langchain.schema import Document
from langchain.vectorstores.base import VST, VectorStore


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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given texts to the store (insert behavior)."""
        raise NotImplementedError()

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
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
        assert index(loader, record_manager, vector_store, delete_mode="full") == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    with patch.object(
        record_manager, "get_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        assert index(loader, record_manager, vector_store, delete_mode="full") == {
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
        assert index(loader, record_manager, vector_store, delete_mode="full") == {
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
        assert index(loader, record_manager, vector_store, delete_mode="full") == {
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
        index(loader, record_manager, vector_store, delete_mode="incremental")

    with pytest.raises(ValueError):
        # Should raise an error because no source id function was specified
        index(
            loader,
            record_manager,
            vector_store,
            delete_mode="incremental",
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
            delete_mode=None,
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
            delete_mode=None,
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
            delete_mode=None,
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
            delete_mode="incremental",
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
            delete_mode="incremental",
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
            delete_mode="incremental",
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

    assert index(loader, record_manager, vector_store, delete_mode="full") == {
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
    assert index(docs, record_manager, vector_store, delete_mode="full") == {
        "num_added": 1,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }
