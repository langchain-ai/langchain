"""Logic for converting internal query language to a valid Milvus query."""
from typing import Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

COMPARATOR_TO_BER = {
    Comparator.EQ: "==",
    Comparator.GT: ">",
    Comparator.GTE: ">=",
    Comparator.LT: "<",
    Comparator.LTE: "<=",
}

UNARY_OPERATORS = [Operator.NOT]


def process_value(value: Union[int, float, str]) -> str:
    # required for comparators involving strings
    if isinstance(value, str):
        # If the value is already a string, add double quotes
        return f'"{value}"'
    else:
        # If the valueis not a string, convert it to a string without double quotes
        return str(value)


class MilvusTranslator(Visitor):
    """Translate Milvus internal query language elements to valid filters."""

    """Subset of allowed logical operators."""
    allowed_operators = [Operator.AND, Operator.NOT, Operator.OR]

    """Subset of allowed logical comparators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        value = func.value
        if isinstance(func, Comparator):
            value = COMPARATOR_TO_BER[func]
        return f"{value}"

    def visit_operation(self, operation: Operation) -> str:
        if operation.operator in UNARY_OPERATORS and len(operation.arguments) == 1:
            operator = self._format_func(operation.operator)
            return operator + "(" + operation.arguments[0].accept(self) + ")"
        elif operation.operator in UNARY_OPERATORS:
            raise ValueError(
                f'"{operation.operator.value}" can have only one argument in Milvus'
            )
        else:
            args = [arg.accept(self) for arg in operation.arguments]
            operator = self._format_func(operation.operator)
            return "(" + (" " + operator + " ").join(args) + ")"

    def visit_comparison(self, comparison: Comparison) -> str:
        comparator = self._format_func(comparison.comparator)
        processed_value = process_value(comparison.value)
        attribute = comparison.attribute

        return "( " + attribute + " " + comparator + " " + processed_value + " )"

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"expr": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
