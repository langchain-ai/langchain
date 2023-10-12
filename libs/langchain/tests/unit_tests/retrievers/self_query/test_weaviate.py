from datetime import date, datetime
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


def test_visit_comparison_integer() -> None:
    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=1)
    expected = {"operator": "GreaterThanEqual", "path": ["foo"], "valueInt": 1}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_number() -> None:
    comp = Comparison(comparator=Comparator.GT, attribute="foo", value=1.4)
    expected = {"operator": "GreaterThan", "path": ["foo"], "valueNumber": 1.4}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_boolean() -> None:
    comp = Comparison(comparator=Comparator.NE, attribute="foo", value=False)
    expected = {"operator": "NotEqual", "path": ["foo"], "valueBoolean": False}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_datetime() -> None:
    comp = Comparison(
        comparator=Comparator.LTE,
        attribute="foo",
        value=datetime(2023, 9, 13, 4, 20, 0),
    )
    expected = {
        "operator": "LessThanEqual",
        "path": ["foo"],
        "valueDate": "2023-09-13T04:20:00Z",
    }
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_date() -> None:
    comp = Comparison(
        comparator=Comparator.LT, attribute="foo", value=date(2023, 9, 13)
    )
    expected = {
        "operator": "LessThan",
        "path": ["foo"],
        "valueDate": "2023-09-13T00:00:00Z",
    }
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="hello"),
            Comparison(
                comparator=Comparator.GTE, attribute="bar", value=date(2023, 9, 13)
            ),
            Comparison(comparator=Comparator.LTE, attribute="abc", value=1.4),
        ],
    )
    expected = {
        "operands": [
            {"operator": "Equal", "path": ["foo"], "valueText": "hello"},
            {
                "operator": "GreaterThanEqual",
                "path": ["bar"],
                "valueDate": "2023-09-13T00:00:00Z",
            },
            {"operator": "LessThanEqual", "path": ["abc"], "valueNumber": 1.4},
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
                    {"path": ["foo"], "operator": "Equal", "valueInt": 2},
                    {"path": ["bar"], "operator": "Equal", "valueText": "baz"},
                ],
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
