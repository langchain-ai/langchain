from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

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
    from qdrant_client.http import models as rest


class QdrantTranslator(Visitor):
    """Translate `Qdrant` internal query language elements to valid filters."""

    allowed_operators = (
        Operator.AND,
        Operator.OR,
        Operator.NOT,
    )
    """Subset of allowed logical operators."""

    allowed_comparators = (
        Comparator.EQ,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
    )
    """Subset of allowed logical comparators."""

    def __init__(self, metadata_key: str):
        self.metadata_key = metadata_key

    def visit_operation(self, operation: Operation) -> rest.Filter:
        try:
            from qdrant_client.http import models as rest
        except ImportError as e:
            raise ImportError(
                "Cannot import qdrant_client. Please install with `pip install "
                "qdrant-client`."
            ) from e

        args = [arg.accept(self) for arg in operation.arguments]
        operator = {
            Operator.AND: "must",
            Operator.OR: "should",
            Operator.NOT: "must_not",
        }[operation.operator]
        return rest.Filter(**{operator: args})

    def visit_comparison(self, comparison: Comparison) -> rest.FieldCondition:
        try:
            from qdrant_client.http import models as rest
        except ImportError as e:
            raise ImportError(
                "Cannot import qdrant_client. Please install with `pip install "
                "qdrant-client`."
            ) from e

        if isinstance(comparison.attribute, VirtualColumnName):
            attribute = comparison.attribute()
        elif isinstance(comparison.attribute, str):
            attribute = comparison.attribute
        else:
            raise TypeError(
                f"Unknown type {type(comparison.attribute)} for `comparison.attribute`!"
            )

        self._validate_func(comparison.comparator)
        attribute = self.metadata_key + "." + attribute
        if comparison.comparator == Comparator.EQ:
            return rest.FieldCondition(
                key=attribute, match=rest.MatchValue(value=comparison.value)
            )
        kwargs = {comparison.comparator.value: comparison.value}
        return rest.FieldCondition(key=attribute, range=rest.Range(**kwargs))

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        try:
            from qdrant_client.http import models as rest
        except ImportError as e:
            raise ImportError(
                "Cannot import qdrant_client. Please install with `pip install "
                "qdrant-client`."
            ) from e

        if structured_query.filter is None:
            kwargs = {}
        else:
            filter = structured_query.filter.accept(self)
            if isinstance(filter, rest.FieldCondition):
                filter = rest.Filter(must=[filter])
            kwargs = {"filter": filter}
        return structured_query.query, kwargs
