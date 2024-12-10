from typing import Dict, Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class OpenSearchTranslator(Visitor):
    """Translate `OpenSearch` internal query domain-specific
    language elements to valid filters."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    ]
    """Subset of allowed logical comparators."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        comp_operator_map = {
            Comparator.EQ: "term",
            Comparator.LT: "lt",
            Comparator.LTE: "lte",
            Comparator.GT: "gt",
            Comparator.GTE: "gte",
            Comparator.CONTAIN: "wildcard",
            Comparator.LIKE: "fuzzy",
            Operator.AND: "must",
            Operator.OR: "should",
            Operator.NOT: "must_not",
        }
        return comp_operator_map[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]

        return {"bool": {self._format_func(operation.operator): args}}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        field = f"metadata.{comparison.attribute}"

        if comparison.comparator in [
            Comparator.LT,
            Comparator.LTE,
            Comparator.GT,
            Comparator.GTE,
        ]:
            if isinstance(comparison.value, dict):
                if "date" in comparison.value:
                    return {
                        "range": {
                            field: {
                                self._format_func(
                                    comparison.comparator
                                ): comparison.value["date"]
                            }
                        }
                    }
            else:
                return {
                    "range": {
                        field: {
                            self._format_func(comparison.comparator): comparison.value
                        }
                    }
                }

        if comparison.comparator == Comparator.LIKE:
            return {
                self._format_func(comparison.comparator): {
                    field: {"value": comparison.value}
                }
            }

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
            kwargs = {"filter": structured_query.filter.accept(self)}

        return structured_query.query, kwargs
