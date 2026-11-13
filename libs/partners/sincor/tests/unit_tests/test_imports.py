"""Unit tests for imports in `langchain_sincor`."""

from langchain_sincor import __all__

EXPECTED_ALL = [
    "SincorAuctionTool",
    "SincorClient",
    "SincorEscrowTool",
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected imports are in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
