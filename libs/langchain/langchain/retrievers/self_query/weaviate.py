from datetime import date, datetime
from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class WeaviateTranslator(Visitor):
    """Translate `Weaviate` internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GTE,
        Comparator.LTE,
        Comparator.LT,
        Comparator.GT,
    ]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        # https://weaviate.io/developers/weaviate/api/graphql/filters
        map_dict = {
            Operator.AND: "And",
            Operator.OR: "Or",
            Comparator.EQ: "Equal",
            Comparator.NE: "NotEqual",
            Comparator.GTE: "GreaterThanEqual",
            Comparator.LTE: "LessThanEqual",
            Comparator.LT: "LessThan",
            Comparator.GT: "GreaterThan",
        }
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {"operator": self._format_func(operation.operator), "operands": args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        value_type = "valueText"
        if isinstance(comparison.value, bool):
            value_type = "valueBoolean"
        elif isinstance(comparison.value, float):
            value_type = "valueNumber"
        elif isinstance(comparison.value, int):
            value_type = "valueInt"
        elif isinstance(comparison.value, datetime) or isinstance(
            comparison.value, date
        ):
            value_type = "valueDate"
            # ISO 8601 timestamp, formatted as RFC3339
            comparison.value = comparison.value.strftime("%Y-%m-%dT%H:%M:%SZ")
        filter = {
            "path": [comparison.attribute],
            "operator": self._format_func(comparison.comparator),
        }
        filter[value_type] = comparison.value
        return filter

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"where_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
