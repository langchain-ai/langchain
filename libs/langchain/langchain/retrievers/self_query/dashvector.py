"""Logic for converting internal query language to a valid DashVector query."""
from typing import Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.chains.query_constructor.schema import VirtualColumnName


class DashvectorTranslator(Visitor):
    """Logic for converting internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.LIKE,
    ]

    map_dict = {
        Operator.AND: " AND ",
        Operator.OR: " OR ",
        Comparator.EQ: " = ",
        Comparator.GT: " > ",
        Comparator.GTE: " >= ",
        Comparator.LT: " < ",
        Comparator.LTE: " <= ",
        Comparator.LIKE: " LIKE ",
    }

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return self.map_dict[func]

    def visit_operation(self, operation: Operation) -> str:
        args = [arg.accept(self) for arg in operation.arguments]
        return self._format_func(operation.operator).join(args)

    def visit_comparison(self, comparison: Comparison) -> str:
        if isinstance(comparison.attribute, VirtualColumnName):
            attribute = comparison.attribute()
        elif isinstance(comparison.attribute, str):
            attribute = comparison.attribute
        else:
            raise TypeError(
                f"Unknown type {type(comparison.attribute)} for `comparison.attribute`!"
            )

        value = comparison.value
        if isinstance(value, str):
            if comparison.comparator == Comparator.LIKE:
                value = f"'%{value}%'"
            else:
                value = f"'{value}'"
        return f"{attribute}{self._format_func(comparison.comparator)}{value}"

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
