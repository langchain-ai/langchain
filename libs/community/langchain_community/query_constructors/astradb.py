"""Logic for converting internal query language to a valid AstraDB query."""
from typing import Dict, Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

MULTIPLE_ARITY_COMPARATORS = [Comparator.IN, Comparator.NIN]


class AstraDBTranslator(Visitor):
    """Translate AstraDB internal query language elements to valid filters."""

    """Subset of allowed logical comparators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.IN,
        Comparator.NIN,
    ]

    """Subset of allowed logical operators."""
    allowed_operators = [Operator.AND, Operator.OR]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        map_dict = {
            Operator.AND: "$and",
            Operator.OR: "$or",
            Comparator.EQ: "$eq",
            Comparator.NE: "$ne",
            Comparator.GTE: "$gte",
            Comparator.LTE: "$lte",
            Comparator.LT: "$lt",
            Comparator.GT: "$gt",
            Comparator.IN: "$in",
            Comparator.NIN: "$nin",
        }
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        if comparison.comparator in MULTIPLE_ARITY_COMPARATORS and not isinstance(
            comparison.value, list
        ):
            comparison.value = [comparison.value]

        comparator = self._format_func(comparison.comparator)
        return {comparison.attribute: {comparator: comparison.value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
