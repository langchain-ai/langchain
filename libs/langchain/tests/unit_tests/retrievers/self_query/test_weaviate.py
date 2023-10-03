from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.weaviate import WeaviateTranslator

DEFAULT_TRANSLATOR = WeaviateTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    expected = {"operator": "Equal", "path": ["foo"], "valueText": "1"}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {
        "operands": [
            {"operator": "Equal", "path": ["foo"], "valueText": 2},
            {"operator": "Equal", "path": ["bar"], "valueText": "baz"},
        ],
        "operator": "And",
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

    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (
        query,
        {"where_filter": {"path": ["foo"], "operator": "Equal", "valueText": "1"}},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )
    expected = (
        query,
        {
            "where_filter": {
                "operator": "And",
                "operands": [
                    {"path": ["foo"], "operator": "Equal", "valueText": 2},
                    {"path": ["bar"], "operator": "Equal", "valueText": "baz"},
                ],
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
