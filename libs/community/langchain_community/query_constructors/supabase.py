from typing import Any, Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


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
        Comparator.LIKE,
    ]
    """Subset of allowed logical comparators."""

    metadata_column = "metadata"

    def _map_comparator(self, comparator: Comparator) -> str:
        """
        Maps Langchain comparator to PostgREST comparator:

        https://postgrest.org/en/stable/references/api/tables_views.html#operators
        """
        postgrest_comparator = {
            Comparator.EQ: "eq",
            Comparator.NE: "neq",
            Comparator.GT: "gt",
            Comparator.GTE: "gte",
            Comparator.LT: "lt",
            Comparator.LTE: "lte",
            Comparator.LIKE: "like",
        }.get(comparator)

        if postgrest_comparator is None:
            raise Exception(
                f"Comparator '{comparator}' is not currently "
                "supported in Supabase Vector"
            )

        return postgrest_comparator

    def _get_json_operator(self, value: Any) -> str:
        if isinstance(value, str):
            return "->>"
        else:
            return "->"

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        return f"{operation.operator.value}({','.join(args)})"

    def visit_comparison(self, comparison: Comparison) -> str:
        if isinstance(comparison.value, list):
            return self.visit_operation(
                Operation(
                    operator=Operator.AND,
                    arguments=[
                        Comparison(
                            comparator=comparison.comparator,
                            attribute=comparison.attribute,
                            value=value,
                        )
                        for value in comparison.value
                    ],
                )
            )

        return ".".join(
            [
                f"{self.metadata_column}{self._get_json_operator(comparison.value)}{comparison.attribute}",
                f"{self._map_comparator(comparison.comparator)}",
                f"{comparison.value}",
            ]
        )

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, Dict[str, str]]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"postgrest_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
