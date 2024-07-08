from typing import Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class DingoDBTranslator(Visitor):
    """Translate `DingoDB` internal query language elements to valid filters."""

    allowed_comparators = (
        Comparator.EQ,
        Comparator.NE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
    )
    """Subset of allowed logical comparators."""
    allowed_operators = (Operator.AND, Operator.OR)
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return f"${func.value}"

    def visit_operation(self, operation: Operation) -> Operation:
        return operation

    def visit_comparison(self, comparison: Comparison) -> Comparison:
        return comparison

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {
                "search_params": {
                    "langchain_expr": structured_query.filter.accept(self)
                }
            }
        return structured_query.query, kwargs
