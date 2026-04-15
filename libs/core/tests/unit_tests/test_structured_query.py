"""Unit tests for Visitor._validate_func in structured_query.py."""

import pytest

from langchain_core.structured_query import (
    Comparator,
    Operator,
    Visitor,
)


class _TestVisitor(Visitor):
    """Minimal concrete visitor for testing."""

    allowed_comparators = (Comparator.EQ,)
    allowed_operators = (Operator.AND,)

    def visit_operation(self, operation):  # type: ignore[override]
        pass

    def visit_comparison(self, comparison):  # type: ignore[override]
        pass

    def visit_structured_query(self, structured_query):  # type: ignore[override]
        pass


def test_validate_func_disallowed_operator_error_message() -> None:
    """Disallowed operator error must say 'operators', not 'comparators'."""
    visitor = _TestVisitor()
    with pytest.raises(ValueError, match="operators are") as exc_info:
        visitor._validate_func(Operator.OR)
    assert "comparators are" not in str(exc_info.value)


def test_validate_func_allowed_operator_passes() -> None:
    """Allowed operator must not raise."""
    visitor = _TestVisitor()
    visitor._validate_func(Operator.AND)  # should not raise


def test_validate_func_disallowed_comparator_error_message() -> None:
    """Disallowed comparator error must say 'comparators'."""
    visitor = _TestVisitor()
    with pytest.raises(ValueError, match="comparators are"):
        visitor._validate_func(Comparator.GT)


def test_validate_func_allowed_comparator_passes() -> None:
    """Allowed comparator must not raise."""
    visitor = _TestVisitor()
    visitor._validate_func(Comparator.EQ)  # should not raise
