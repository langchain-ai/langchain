from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.deeplake import DeepLakeTranslator

DEFAULT_TRANSLATOR = DeepLakeTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    expected = "(metadata['foo'] < 1 or metadata['foo'] < 2)"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=["1", "2"]),
        ],
    )
    expected = (
        "(metadata['foo'] < 2 and metadata['bar'] == 'baz' "
        "and (metadata['abc'] < 1 or metadata['abc'] < 2))"
    )
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (
        query,
        {"tql": "SELECT * WHERE (metadata['foo'] < 1 or metadata['foo'] < 2)"},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=["1", "2"]),
        ],
    )
    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )
    expected = (
        query,
        {
            "tql": "SELECT * WHERE "
            "(metadata['foo'] < 2 and metadata['bar'] == 'baz' and "
            "(metadata['abc'] < 1 or metadata['abc'] < 2))"
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
