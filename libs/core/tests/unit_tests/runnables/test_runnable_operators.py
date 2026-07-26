"""Tests for ``Runnable.__or__`` / ``Runnable.__ror__`` operator protocol.

Regression tests for https://github.com/langchain-ai/langchain/issues/39075:
when the other operand is not ``Runnable``-like, the dunder must return
``NotImplemented`` (per the Python data model) so the interpreter can try the
other operand's reflected operator, instead of raising ``TypeError`` directly.
"""

import pytest

from langchain_core.runnables import RunnableLambda


class _ForeignRor:
    """Third-party object that defines the reflected ``__ror__``."""

    def __ror__(self, other: object) -> tuple[str, str]:
        return ("foreign_ror", type(other).__name__)


class _ForeignOr:
    """Third-party object that defines ``__or__``."""

    def __or__(self, other: object) -> tuple[str, str]:
        return ("foreign_or", type(other).__name__)


def test_or_defers_to_foreign_ror() -> None:
    chain = RunnableLambda(lambda x: x * 2)
    assert chain | _ForeignRor() == ("foreign_ror", "RunnableLambda")


def test_ror_defers_to_foreign_or() -> None:
    chain = RunnableLambda(lambda x: x * 2)
    assert _ForeignOr() | chain == ("foreign_or", "RunnableLambda")


def test_or_returns_not_implemented_for_unsupported() -> None:
    chain = RunnableLambda(lambda x: x * 2)
    assert chain.__or__(42) is NotImplemented
    assert chain.__ror__(42) is NotImplemented


def test_or_still_raises_for_genuinely_unsupported_operand() -> None:
    chain = RunnableLambda(lambda x: x * 2)
    # neither operand handles the op -> the interpreter raises TypeError
    with pytest.raises(TypeError):
        _ = chain | 42
