"""Test `langchain_requesty` public API surface."""

from langchain_requesty import __all__

EXPECTED_ALL = ["__version__", "ChatRequesty"]


def test_all_imports() -> None:
    """Verify that `__all__` exports match the expected public API."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
