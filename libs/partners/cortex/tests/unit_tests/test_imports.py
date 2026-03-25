"""Smoke-tests for public API surface of langchain-cortex."""

from langchain_cortex import __all__

EXPECTED_ALL = [
    "CortexChatMessageHistory",
    "CortexMemory",
]


def test_all_exports() -> None:
    """__all__ must list exactly the expected public symbols."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_chat_history_importable() -> None:
    """CortexChatMessageHistory must be importable from the top-level package."""
    from langchain_cortex import CortexChatMessageHistory  # noqa: F401


def test_memory_importable() -> None:
    """CortexMemory must be importable from the top-level package."""
    from langchain_cortex import CortexMemory  # noqa: F401
