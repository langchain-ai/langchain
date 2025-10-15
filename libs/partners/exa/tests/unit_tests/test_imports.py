"""Unit tests for imports in `langchain_exa`."""

from langchain_exa import __all__  # type: ignore[import-not-found, import-not-found]

EXPECTED_ALL = [
    "ExaSearchResults",
    "ExaSearchRetriever",
    "HighlightsContentsOptions",
    "TextContentsOptions",
    "ExaFindSimilarResults",
]


def test_all_imports() -> None:
    """Test that all expected imports are in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
