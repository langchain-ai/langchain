"""Tests for structured query visitor validation."""

import pytest

from langchain_core.structured_query import Comparator, Operator, Visitor


class _TestVisitor(Visitor):
    allowed_comparators = (Comparator.EQ,)
    allowed_operators = (Operator.AND,)

    def visit_operation(self, operation):  # noqa: ANN001, ARG002
        return None

    def visit_comparison(self, comparison):  # noqa: ANN001, ARG002
        return None

    def visit_structured_query(self, structured_query):  # noqa: ANN001, ARG002
        return None


def test_validate_func_operator_error_message() -> None:
    visitor = _TestVisitor()
    with pytest.raises(ValueError, match="Allowed operators are"):
        visitor._validate_func(Operator.OR)
