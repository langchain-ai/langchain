from typing import Dict, Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class ElasticsearchTranslator(Visitor):
    """Translate `Elasticsearch` internal query language elements to valid filters."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    ]
    """Subset of allowed logical comparators."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        map_dict = {
            Operator.OR: "should",
            Operator.NOT: "must_not",
            Operator.AND: "must",
            Comparator.EQ: "term",
            Comparator.GT: "gt",
            Comparator.GTE: "gte",
            Comparator.LT: "lt",
            Comparator.LTE: "lte",
            Comparator.CONTAIN: "match",
            Comparator.LIKE: "match",
        }
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]

        return {"bool": {self._format_func(operation.operator): args}}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        # ElasticsearchStore filters require to target
        # the metadata object field
        field = f"metadata.{comparison.attribute}"

        is_range_comparator = comparison.comparator in [
            Comparator.GT,
            Comparator.GTE,
            Comparator.LT,
            Comparator.LTE,
        ]

        if is_range_comparator:
            value = comparison.value
            if isinstance(comparison.value, dict) and "date" in comparison.value:
                value = comparison.value["date"]
            return {"range": {field: {self._format_func(comparison.comparator): value}}}

        if comparison.comparator == Comparator.CONTAIN:
            return {
                self._format_func(comparison.comparator): {
                    field: {"query": comparison.value}
                }
            }

        if comparison.comparator == Comparator.LIKE:
            return {
                self._format_func(comparison.comparator): {
                    field: {"query": comparison.value, "fuzziness": "AUTO"}
                }
            }

        # we assume that if the value is a string,
        # we want to use the keyword field
        field = f"{field}.keyword" if isinstance(comparison.value, str) else field

        if isinstance(comparison.value, dict):
            if "date" in comparison.value:
                comparison.value = comparison.value["date"]

        return {self._format_func(comparison.comparator): {field: comparison.value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": [structured_query.filter.accept(self)]}
        return structured_query.query, kwargs
