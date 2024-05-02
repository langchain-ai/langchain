from typing import Dict, Tuple

import pytest as pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.pgvector import PGVectorTranslator

DEFAULT_TRANSLATOR = PGVectorTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = {"foo": {"lt": 1}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


@pytest.mark.skip("Not implemented")
def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    expected = {
        "foo": {"lt": 2},
        "bar": {"eq": "baz"},
        "abc": {"gt": 2.0},
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

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (query, {"filter": {"foo": {"lt": 1}}})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
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
                "and": [
                    {"foo": {"lt": 2}},
                    {"bar": {"eq": "baz"}},
                    {"abc": {"gt": 2.0}},
                ]
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
