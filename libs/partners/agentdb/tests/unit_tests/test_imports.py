"""Unit tests for imports in `langchain_agentdb`."""

from langchain_agentdb import __all__  # type: ignore[import-not-found]

EXPECTED_ALL = [
    "AgentDBRetriever",
]


def test_all_imports() -> None:
    """Test that all expected symbols are exported in __all__."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
