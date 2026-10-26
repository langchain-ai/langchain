"""Unit tests for the Exa retriever."""

from types import SimpleNamespace
from typing import Any

from langchain_exa.retrievers import _get_metadata


def _make_result(**optional_metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(
        title="Example",
        url="https://example.com",
        id="result-1",
        score=0.95,
        published_date="2024-01-01",
        author="Author",
        **optional_metadata,
    )


def test_get_metadata_omits_missing_optional_attributes() -> None:
    """Test that missing optional result attributes are omitted."""
    assert _get_metadata(_make_result()) == {
        "title": "Example",
        "url": "https://example.com",
        "id": "result-1",
        "score": 0.95,
        "published_date": "2024-01-01",
        "author": "Author",
    }


def test_get_metadata_includes_available_optional_attributes() -> None:
    """Test that available optional attributes are retained independently."""
    metadata = _get_metadata(
        _make_result(highlights=["Excerpt"], summary="A short summary")
    )

    assert metadata["highlights"] == ["Excerpt"]
    assert metadata["summary"] == "A short summary"
    assert "highlight_scores" not in metadata
