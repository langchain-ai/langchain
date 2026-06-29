"""Unit tests for Visitor._validate_func error messages."""

import pytest

from langchain_core.structured_query import Comparator, Operator, Visitor


class _MinimalVisitor(Visitor):
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
    visitor = _MinimalVisitor()
    with pytest.raises(ValueError, match="operator") as exc_info:
        visitor._validate_func(Operator.OR)
    assert "comparators" not in str(exc_info.value), (
        "Error message for a disallowed operator incorrectly says 'comparators'"
    )


def test_validate_func_disallowed_comparator_error_message() -> None:
    """Disallowed comparator error must say 'comparators'."""
    visitor = _MinimalVisitor()
    with pytest.raises(ValueError, match="comparator"):
        visitor._validate_func(Comparator.GT)


def test_validate_func_allowed_operator_does_not_raise() -> None:
    visitor = _MinimalVisitor()
    visitor._validate_func(Operator.AND)  # should not raise


def test_validate_func_allowed_comparator_does_not_raise() -> None:
    visitor = _MinimalVisitor()
    visitor._validate_func(Comparator.EQ)  # should not raise
