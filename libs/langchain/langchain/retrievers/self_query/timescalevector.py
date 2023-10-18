from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.chains.query_constructor.schema import VirtualColumnName

if TYPE_CHECKING:
    from timescale_vector import client


class TimescaleVectorTranslator(Visitor):
    """Translate the internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ]

    COMPARATOR_MAP = {
        Comparator.EQ: "==",
        Comparator.GT: ">",
        Comparator.GTE: ">=",
        Comparator.LT: "<",
        Comparator.LTE: "<=",
    }

    OPERATOR_MAP = {Operator.AND: "AND", Operator.OR: "OR", Operator.NOT: "NOT"}

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        if isinstance(func, Operator):
            value = self.OPERATOR_MAP[func.value]  # type: ignore
        elif isinstance(func, Comparator):
            value = self.COMPARATOR_MAP[func.value]  # type: ignore
        return f"{value}"

    def visit_operation(self, operation: Operation) -> client.Predicates:
        try:
            from timescale_vector import client
        except ImportError as e:
            raise ImportError(
                "Cannot import timescale-vector. Please install with `pip install "
                "timescale-vector`."
            ) from e
        args = [arg.accept(self) for arg in operation.arguments]
        return client.Predicates(*args, operator=self._format_func(operation.operator))

    def visit_comparison(self, comparison: Comparison) -> client.Predicates:
        if isinstance(comparison.attribute, VirtualColumnName):
            attribute = comparison.attribute()
        elif isinstance(comparison.attribute, str):
            attribute = comparison.attribute
        else:
            raise TypeError(
                f"Unknown type {type(comparison.attribute)} for `comparison.attribute`!"
            )
        try:
            from timescale_vector import client
        except ImportError as e:
            raise ImportError(
                "Cannot import timescale-vector. Please install with `pip install "
                "timescale-vector`."
            ) from e
        return client.Predicates(
            (
                attribute,
                self._format_func(comparison.comparator),
                comparison.value,
            )
        )

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"predicates": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
