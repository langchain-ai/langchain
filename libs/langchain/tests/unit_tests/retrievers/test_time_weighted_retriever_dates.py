"""Additional tests for date handling in the time-weighted retriever."""

from datetime import datetime

from langchain_core.documents import Document

from langchain_classic.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)


def _make_retriever() -> TimeWeightedVectorStoreRetriever:
    """Create a minimal retriever instance suitable for unit tests.

    We don't need a working vector store here because the tests exercise only the
    internal date-handling helpers, not similarity search.
    """

    class _DummyVectorStore:  # type: ignore[too-many-instance-attributes]
        """Minimal stub satisfying the vector store interface at type-check time."""

        # The retriever only needs the attribute for type checking; it is never used
        # in these tests.
        def __init__(self) -> None:  # pragma: no cover - trivial
            ...

    return TimeWeightedVectorStoreRetriever(
        vectorstore=_DummyVectorStore(),  # type: ignore[arg-type]
        memory_stream=[],
    )


def test_document_get_date_accepts_iso_string() -> None:
    """ISO 8601 strings in metadata should be converted to ``datetime``."""
    retriever = _make_retriever()
    ts = datetime(2024, 1, 1, 12, 0, 0)
    doc = Document(
        page_content="foo",
        metadata={"last_accessed_at": ts.isoformat()},
    )

    result = retriever._document_get_date("last_accessed_at", doc)

    assert isinstance(result, datetime)
    assert result == ts


def test_document_get_date_accepts_numeric_timestamp() -> None:
    """Numeric timestamps in metadata should be converted to ``datetime``."""
    retriever = _make_retriever()
    ts = datetime(2024, 1, 1, 12, 0, 0)
    numeric_ts = ts.timestamp()
    doc = Document(
        page_content="foo",
        metadata={"last_accessed_at": numeric_ts},
    )

    result = retriever._document_get_date("last_accessed_at", doc)

    assert isinstance(result, datetime)
    # Use timestamp comparison to avoid timezone/localtime edge cases.
    assert result.timestamp() == numeric_ts

