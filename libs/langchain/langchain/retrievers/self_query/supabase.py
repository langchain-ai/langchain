from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from urllib.parse import quote
from numbers import Number


class SupabaseVectorTranslator(Visitor):
    """Translate Langchain filters to Supabase PostgREST filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    ]
    """Subset of allowed logical comparators."""

    metadata_column = "metadata"

    def _format_value(self, value, comparator: Comparator):
        if comparator == Comparator.CONTAIN:
            return f"%{value}%"

        return value

    def _map_comparator(self, comparator: str):
        """
        Maps Langchain comparator to PostgREST comparator:

        https://postgrest.org/en/stable/references/api/tables_views.html#operators
        """
        return {
            Comparator.EQ: "eq",
            Comparator.NE: "neq",
            Comparator.GT: "gt",
            Comparator.GTE: "gte",
            Comparator.LT: "lt",
            Comparator.LTE: "lte",
            Comparator.CONTAIN: "like",
            Comparator.LIKE: "like",
        }.get(comparator, comparator)

    def _get_json_operator(self, value):
        if isinstance(value, str):
            return "->>"
        else:
            return "->"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return f"{operation.operator}({','.join(args)})"

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return ".".join(
            [
                f"{self.metadata_column}{self._get_json_operator(comparison.value)}{comparison.attribute}",
                f"{self._map_comparator(comparison.comparator)}",
                f"{self._format_value(comparison.value, comparison.comparator)}",
            ]
        )

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"postgrest_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
