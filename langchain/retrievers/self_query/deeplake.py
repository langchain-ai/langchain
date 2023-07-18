"""Logic for converting internal query language to a valid Chroma query."""
from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
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
}


def can_cast_to_float(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


class DeepLakeTranslator(Visitor):
    """Logic for converting internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
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
            value = OPERATOR_TO_TQL[func.value]
        elif isinstance(func, Comparator):
            value = COMPARATOR_TO_TQL[func.value]
        return f"{value}"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        operator = self._format_func(operation.operator)
        return "(" + (" " + operator + " ").join(args) + ")"

    def visit_comparison(self, comparison: Comparison) -> Dict:
        comparator = self._format_func(comparison.comparator)
        value = comparison.value
        if not can_cast_to_float(comparison.value):
            value = f"'{value}'"
        return f"metadata['{comparison.attribute}'] {comparator} {value}"

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            tqL = f"SELECT * WHERE {structured_query.filter.accept(self)}"
            kwargs = {"tql": tqL}
        return structured_query.query, kwargs
