"""Test `langchain_openrouter` public API surface."""

from langchain_openrouter import __all__

EXPECTED_ALL = [
    "ChatOpenRouter",
]


def test_all_imports() -> None:
    """Verify that __all__ exports match the expected public API."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
