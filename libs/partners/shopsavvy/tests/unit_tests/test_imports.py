"""Unit tests for imports in ``langchain_shopsavvy``."""

from langchain_shopsavvy import __all__  # type: ignore[import-not-found]

EXPECTED_ALL = [
    "ShopSavvyPriceComparison",
    "ShopSavvyPriceHistory",
    "ShopSavvyProductSearch",
    "ShopSavvyRetriever",
]


def test_all_imports() -> None:
    """Test that all expected imports are in ``__all__``."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
