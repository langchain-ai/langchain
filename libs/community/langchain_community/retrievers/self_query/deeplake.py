"""Logic for converting internal query language to a valid Chroma query."""
from typing import Tuple, Union

from langchain_core.sql_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

COMPARATOR_TO_TQL = {
    Comparator.EQ: "==",
    Comparator.GT: ">",
    Comparator.GTE: ">=",
    Comparator.LT: "<",
    Comparator.LTE: "<=",
}


OPERATOR_TO_TQL = {
    Operator.AND: "and",
    Operator.OR: "or",
    Operator.NOT: "NOT",
}


def can_cast_to_float(string: str) -> bool:
    """Check if a string can be cast to a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False


class DeepLakeTranslator(Visitor):
    """Translate `DeepLake` internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ]
    """Subset of allowed logical comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        if isinstance(func, Operator):
            value = OPERATOR_TO_TQL[func.value]  # type: ignore
        elif isinstance(func, Comparator):
            value = COMPARATOR_TO_TQL[func.value]  # type: ignore
        return f"{value}"

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        operator = self._format_func(operation.operator)
        return "(" + (" " + operator + " ").join(args) + ")"

    def visit_comparison(self, comparison: Comparison) -> str:
        comparator = self._format_func(comparison.comparator)
        values = comparison.value
        if isinstance(values, list):
            tql = []
            for value in values:
                comparison.value = value
                tql.append(self.visit_comparison(comparison))

            return "(" + (" or ").join(tql) + ")"

        if not can_cast_to_float(comparison.value):
            values = f"'{values}'"
        return f"metadata['{comparison.attribute}'] {comparator} {values}"

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            tqL = f"SELECT * WHERE {structured_query.filter.accept(self)}"
            kwargs = {"tql": tqL}
        return structured_query.query, kwargs
