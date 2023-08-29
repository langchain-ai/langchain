from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator

DEFAULT_TRANSLATOR = ChromaTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    expected = {"foo": {"$lt": ["1", "2"]}}
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
    expected = {
        "$and": [
            {"foo": {"$lt": 2}},
            {"bar": {"$eq": "baz"}},
            {"abc": {"$lt": ["1", "2"]}},
        ]
    }
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
    expected = (
        query,
        {"filter": {"foo": {"$lt": ["1", "2"]}}},
    )
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
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
            "filter": {
                "$and": [
                    {"foo": {"$lt": 2}},
                    {"bar": {"$eq": "baz"}},
                    {"abc": {"$lt": ["1", "2"]}},
                ]
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
