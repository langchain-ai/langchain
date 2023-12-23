from datetime import datetime
from typing import Any, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class PathwayTranslator(Visitor):
    """Translate `Pathway` internal query language elements to valid filters."""

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
    ]
    """Subset of allowed logical comparators."""

    map_dict = {
        Operator.AND: " && ",
        Operator.OR: " || ",
        Comparator.EQ: "==",
        Comparator.NE: "!=",
        Comparator.GTE: ">=",
        Comparator.LTE: "<=",
        Comparator.LT: "<",
        Comparator.GT: ">",
    }

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return f"`{str(value).lower()}`"
        elif isinstance(value, dict) and value.get("type") == "date":
            date = datetime.strptime(value["date"], "%Y-%m-%d")
            # convert date to timestamp as JMESPath does not support dates
            timestamp = int(date.timestamp())
            return f"`{timestamp}`"
        else:
            return f"`{value}`"

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        return self.map_dict[operation.operator].join(args)

    def visit_comparison(self, comparison: Comparison) -> str:
        if comparison.comparator == Comparator.CONTAIN:
            return (
                f"contains({comparison.attribute}, "
                f"{self._format_value(comparison.value)})"
            )
        else:
            return (
                f"{comparison.attribute} "
                f"{self.map_dict[comparison.comparator]} "
                f"{self._format_value(comparison.value)}"
            )

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"metadata_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
