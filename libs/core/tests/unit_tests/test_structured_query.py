"""Unit tests for `langchain_core.structured_query`."""

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
    allowed_comparators = (Comparator.EQ,)
    allowed_operators = (Operator.AND,)

    def visit_operation(self, operation: Operation) -> None:
        return None

    def visit_comparison(self, comparison: Comparison) -> None:
        return None

    def visit_structured_query(self, structured_query: StructuredQuery) -> None:
        return None


def test_validate_func_disallowed_operator_message_mentions_operators() -> None:
    visitor = _StubVisitor()
    with pytest.raises(ValueError, match="Allowed operators are") as exc_info:
        visitor._validate_func(Operator.OR)
    assert "comparators" not in str(exc_info.value)


def test_validate_func_disallowed_comparator_message_mentions_comparators() -> None:
    visitor = _StubVisitor()
    with pytest.raises(ValueError, match="Allowed comparators are"):
        visitor._validate_func(Comparator.NE)
