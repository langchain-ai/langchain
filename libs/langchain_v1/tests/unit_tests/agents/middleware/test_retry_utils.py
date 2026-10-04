"""Tests for `should_retry_exception` retry_on handling."""

import pytest

from langchain.agents.middleware._retry import should_retry_exception


def test_bare_class_non_match_returns_false() -> None:
    """A bare exception class that does not match must not be invoked as a predicate."""
    assert should_retry_exception(KeyError("k"), ValueError) is False


def test_bare_class_match_returns_true() -> None:
    assert should_retry_exception(TimeoutError("t"), TimeoutError) is True


def test_tuple_of_classes() -> None:
    assert should_retry_exception(KeyError("k"), (ValueError, TimeoutError)) is False
    assert should_retry_exception(TimeoutError("t"), (ValueError, TimeoutError)) is True


def test_subclass_matching() -> None:
    assert should_retry_exception(ConnectionError("c"), OSError) is True


def test_callable_predicate_still_works() -> None:
    assert should_retry_exception(KeyError("k"), lambda exc: True) is True
    assert should_retry_exception(KeyError("k"), lambda exc: isinstance(exc, KeyError)) is True
    assert should_retry_exception(KeyError("k"), lambda exc: False) is False


def test_bare_class_returns_bool_not_instance() -> None:
    """The bare-class branch must return a real bool, never an exception instance."""
    result = should_retry_exception(KeyError("k"), ValueError)
    assert isinstance(result, bool)
