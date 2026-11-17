"""Unit tests for imports in langchain_convexlake."""

from langchain_convexlake import __all__

EXPECTED_ALL = [
    "ConvexLakeCatalog",
    "ConvexLakeCandles",
    "ConvexLakeLeaderboard",
    "ConvexLakeLeaderboardTrades",
    "ConvexLakeOrderbookSummary",
    "ConvexLakeSurface",
    "ConvexLakeTrades",
    "__version__",
    "get_tools",
]


def test_all_imports() -> None:
    """Test that all expected symbols are in __all__."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
