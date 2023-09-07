from typing import Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


def process_value(value: Union[int, float, str]) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    else:
        return str(value)


class VectaraTranslator(Visitor):
    """Translate `Vectara` internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ]
    """Subset of allowed logical comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        map_dict = {
            Operator.AND: " and ",
            Operator.OR: " or ",
            Comparator.EQ: "=",
            Comparator.NE: "!=",
            Comparator.GT: ">",
            Comparator.GTE: ">=",
            Comparator.LT: "<",
            Comparator.LTE: "<=",
        }
        self._validate_func(func)
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        operator = self._format_func(operation.operator)
        return "( " + operator.join(args) + " )"

    def visit_comparison(self, comparison: Comparison) -> str:
        comparator = self._format_func(comparison.comparator)
        processed_value = process_value(comparison.value)
        attribute = comparison.attribute
        return (
            "( " + "doc." + attribute + " " + comparator + " " + processed_value + " )"
        )

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
