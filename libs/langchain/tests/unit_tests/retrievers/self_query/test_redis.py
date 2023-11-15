from typing import Dict, Tuple

import pytest

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.vectorstores.redis.filters import (
    RedisFilterExpression,
    RedisNum,
    RedisTag,
    RedisText,
)
from langchain.vectorstores.redis.schema import (
    NumericFieldSchema,
    RedisModel,
    TagFieldSchema,
    TextFieldSchema,
)


@pytest.fixture
def translator() -> RedisTranslator:
    schema = RedisModel(
        text=[TextFieldSchema(name="bar")],
        numeric=[NumericFieldSchema(name="foo")],
        tag=[TagFieldSchema(name="tag")],
    )
    return RedisTranslator(schema)


@pytest.mark.parametrize(
    ("comp", "expected"),
    [
        (
            Comparison(comparator=Comparator.LT, attribute="foo", value=1),
            RedisNum("foo") < 1,
        ),
        (
            Comparison(comparator=Comparator.LIKE, attribute="bar", value="baz*"),
            RedisText("bar") % "baz*",
        ),
        (
            Comparison(
                comparator=Comparator.CONTAIN, attribute="tag", value=["blue", "green"]
            ),
            RedisTag("tag") == ["blue", "green"],
        ),
    ],
)
def test_visit_comparison(
    translator: RedisTranslator, comp: Comparison, expected: RedisFilterExpression
) -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = RedisNum("foo") < 1
    actual = translator.visit_comparison(comp)
    assert str(expected) == str(actual)


def test_visit_operation(translator: RedisTranslator) -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.EQ, attribute="tag", value="high"),
        ],
    )
    expected = (RedisNum("foo") < 2) & (
        (RedisText("bar") == "baz") & (RedisTag("tag") == "high")
    )
    actual = translator.visit_operation(op)
    assert str(expected) == str(actual)


def test_visit_structured_query_no_filter(translator: RedisTranslator) -> None:
    query = "What is the capital of France?"

    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = translator.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_comparison(translator: RedisTranslator) -> None:
    query = "What is the capital of France?"
    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=2)
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected_filter = RedisNum("foo") >= 2
    actual_query, actual_filter = translator.visit_structured_query(structured_query)
    assert actual_query == query
    assert str(actual_filter["filter"]) == str(expected_filter)


def test_visit_structured_query_operation(translator: RedisTranslator) -> None:
    query = "What is the capital of France?"
    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.CONTAIN, attribute="bar", value="baz"),
        ],
    )
    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )
    expected_filter = (RedisNum("foo") == 2) | (RedisText("bar") == "baz")
    actual_query, actual_filter = translator.visit_structured_query(structured_query)
    assert actual_query == query
    assert str(actual_filter["filter"]) == str(expected_filter)
