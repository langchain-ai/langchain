from collections import ChainMap
from itertools import chain
from typing import Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

_COMPARATOR_TO_SYMBOL = {
    Comparator.EQ: "",
    Comparator.GT: " >",
    Comparator.GTE: " >=",
    Comparator.LT: " <",
    Comparator.LTE: " <=",
    Comparator.IN: "",
    Comparator.LIKE: " LIKE",
}


class DatabricksVectorSearchTranslator(Visitor):
    """Translate `Databricks vector search` internal query language elements to
    valid filters."""

    """Subset of allowed logical operators."""
    allowed_operators = [Operator.AND, Operator.NOT, Operator.OR]

    """Subset of allowed logical comparators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.IN,
        Comparator.LIKE,
    ]

    def _visit_and_operation(self, operation: Operation) -> Dict:
        return dict(ChainMap(*[arg.accept(self) for arg in operation.arguments]))

    def _visit_or_operation(self, operation: Operation) -> Dict:
        filter_args = [arg.accept(self) for arg in operation.arguments]
        flattened_args = list(
            chain.from_iterable(filter_arg.items() for filter_arg in filter_args)
        )
        return {
            " OR ".join(key for key, _ in flattened_args): [
                value for _, value in flattened_args
            ]
        }

    def _visit_not_operation(self, operation: Operation) -> Dict:
        if len(operation.arguments) > 1:
            raise ValueError(
                f'"{operation.operator.value}" can have only one argument '
                f"in Databricks vector search"
            )
        filter_arg = operation.arguments[0].accept(self)
        return {
            f"{colum_with_bool_expression} NOT": value
            for colum_with_bool_expression, value in filter_arg.items()
        }

    def visit_operation(self, operation: Operation) -> Dict:
        self._validate_func(operation.operator)
        if operation.operator == Operator.AND:
            return self._visit_and_operation(operation)
        elif operation.operator == Operator.OR:
            return self._visit_or_operation(operation)
        elif operation.operator == Operator.NOT:
            return self._visit_not_operation(operation)

    def visit_comparison(self, comparison: Comparison) -> Dict:
        self._validate_func(comparison.comparator)
        comparator_symbol = _COMPARATOR_TO_SYMBOL[comparison.comparator]
        return {f"{comparison.attribute}{comparator_symbol}": comparison.value}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filters": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
