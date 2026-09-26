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


class DummyVisitor(Visitor):
    allowed_operators = (Operator.AND,)
    allowed_comparators = (Comparator.EQ,)

    def visit_operation(self, _operation: Operation) -> Any:
        return "operation"

    def visit_comparison(self, _comparison: Comparison) -> Any:
        return "comparison"

    def visit_structured_query(self, _structured_query: StructuredQuery) -> Any:
        return "structured_query"


def test_visitor_validate_func_disallowed_operator() -> None:
    visitor = DummyVisitor()
    # Allowed operator should pass without raising
    visitor._validate_func(Operator.AND)

    # Disallowed operator should raise ValueError mentioning "Allowed operators are"
    with pytest.raises(ValueError, match=r"Allowed operators are"):
        visitor._validate_func(Operator.OR)


def test_visitor_validate_func_disallowed_comparator() -> None:
    visitor = DummyVisitor()
    # Allowed comparator should pass without raising
    visitor._validate_func(Comparator.EQ)

    # Disallowed comparator should raise ValueError mentioning "Allowed comparators are"
    with pytest.raises(ValueError, match=r"Allowed comparators are"):
        visitor._validate_func(Comparator.GT)
