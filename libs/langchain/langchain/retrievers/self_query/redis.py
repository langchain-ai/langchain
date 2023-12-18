from __future__ import annotations

from typing import Any, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.vectorstores.redis import Redis
from langchain.vectorstores.redis.filters import (
    RedisFilterExpression,
    RedisFilterField,
    RedisFilterOperator,
    RedisNum,
    RedisTag,
    RedisText,
)
from langchain.vectorstores.redis.schema import RedisModel

_COMPARATOR_TO_BUILTIN_METHOD = {
    Comparator.EQ: "__eq__",
    Comparator.NE: "__ne__",
    Comparator.LT: "__lt__",
    Comparator.GT: "__gt__",
    Comparator.LTE: "__le__",
    Comparator.GTE: "__ge__",
    Comparator.CONTAIN: "__eq__",
    Comparator.LIKE: "__mod__",
}


class RedisTranslator(Visitor):
    """Visitor for translating structured queries to Redis filter expressions."""

    allowed_comparators = (
        Comparator.EQ,
        Comparator.NE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    )
    """Subset of allowed logical comparators."""
    allowed_operators = (Operator.AND, Operator.OR)
    """Subset of allowed logical operators."""

    def __init__(self, schema: RedisModel) -> None:
        self._schema = schema

    def _attribute_to_filter_field(self, attribute: str) -> RedisFilterField:
        if attribute in [tf.name for tf in self._schema.text]:
            return RedisText(attribute)
        elif attribute in [tf.name for tf in self._schema.tag or []]:
            return RedisTag(attribute)
        elif attribute in [tf.name for tf in self._schema.numeric or []]:
            return RedisNum(attribute)
        else:
            raise ValueError(
                f"Invalid attribute {attribute} not in vector store schema. Schema is:"
                f"\n{self._schema.as_dict()}"
            )

    def visit_comparison(self, comparison: Comparison) -> RedisFilterExpression:
        filter_field = self._attribute_to_filter_field(comparison.attribute)
        comparison_method = _COMPARATOR_TO_BUILTIN_METHOD[comparison.comparator]
        return getattr(filter_field, comparison_method)(comparison.value)

    def visit_operation(self, operation: Operation) -> Any:
        left = operation.arguments[0].accept(self)
        if len(operation.arguments) > 2:
            right = self.visit_operation(
                Operation(
                    operator=operation.operator, arguments=operation.arguments[1:]
                )
            )
        else:
            right = operation.arguments[1].accept(self)
        redis_operator = (
            RedisFilterOperator.OR
            if operation.operator == Operator.OR
            else RedisFilterOperator.AND
        )
        return RedisFilterExpression(operator=redis_operator, left=left, right=right)

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs

    @classmethod
    def from_vectorstore(cls, vectorstore: Redis) -> RedisTranslator:
        return cls(vectorstore._schema)
