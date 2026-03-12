"""Test imports for `langchain_minimax` package."""

from langchain_minimax import __all__

EXPECTED_ALL = ["ChatMiniMax", "__version__"]


def test_all_imports() -> None:
    """Test that all expected symbols are exported."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
