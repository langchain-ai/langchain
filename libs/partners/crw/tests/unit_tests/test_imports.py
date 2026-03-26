"""Unit tests for imports in `langchain_crw`."""

from langchain_crw import __all__  # type: ignore[import-not-found, import-not-found]

EXPECTED_ALL = [
    "CrwLoader",
]


def test_all_imports() -> None:
    """Test that all expected imports are in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
