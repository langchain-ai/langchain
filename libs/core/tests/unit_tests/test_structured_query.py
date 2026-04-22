"""Tests for `langchain_core.structured_query`."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class _StubVisitor(Visitor):
    """Minimal `Visitor` implementation for exercising `_validate_func`."""

    def visit_operation(self, operation: Operation) -> Any:
        return operation

    def visit_comparison(self, comparison: Comparison) -> Any:
        return comparison

    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        return structured_query


def test_validate_func_disallowed_operator_error_message() -> None:
    """Disallowed operators should surface an operator-specific error message.

    Regression test: previously the message talked about `comparators` when an
    operator was rejected, which was misleading when debugging filter queries.
    """
    visitor = _StubVisitor()
    visitor.allowed_operators = [Operator.AND]

    with pytest.raises(ValueError, match=r"Allowed operators are"):
        visitor._validate_func(Operator.NOT)


def test_validate_func_disallowed_comparator_error_message() -> None:
    """Disallowed comparators should keep their comparator-specific message."""
    visitor = _StubVisitor()
    visitor.allowed_comparators = [Comparator.EQ]

    with pytest.raises(ValueError, match=r"Allowed comparators are"):
        visitor._validate_func(Comparator.LT)


def test_validate_func_allows_permitted_operator() -> None:
    visitor = _StubVisitor()
    visitor.allowed_operators = [Operator.AND, Operator.OR]

    visitor._validate_func(Operator.AND)


def test_validate_func_allows_permitted_comparator() -> None:
    visitor = _StubVisitor()
    visitor.allowed_comparators = [Comparator.EQ, Comparator.NE]

    visitor._validate_func(Comparator.EQ)
