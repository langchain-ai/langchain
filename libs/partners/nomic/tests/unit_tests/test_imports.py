"""Unit tests for imports in Nomic partner integration."""

from langchain_nomic import __all__

EXPECTED_ALL = [
    "NomicEmbeddings",
]


def test_all_imports() -> None:
    """Test that all expected imports are present in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
