from typing import Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.astradb import AstraDBTranslator

DEFAULT_TRANSLATOR = AstraDBTranslator()


def test_visit_comparison_lt() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="qty", value=20)
    expected = {"qty": {"$lt": 20}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_eq() -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="qty", value=10)
    expected = {"qty": {"$eq": 10}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_ne() -> None:
    comp = Comparison(comparator=Comparator.NE, attribute="name", value="foo")
    expected = {"name": {"$ne": "foo"}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_in() -> None:
    comp = Comparison(comparator=Comparator.IN, attribute="name", value="foo")
    expected = {"name": {"$in": ["foo"]}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_nin() -> None:
    comp = Comparison(comparator=Comparator.NIN, attribute="name", value="foo")
    expected = {"name": {"$nin": ["foo"]}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.GTE, attribute="qty", value=10),
            Comparison(comparator=Comparator.LTE, attribute="qty", value=20),
            Comparison(comparator=Comparator.EQ, attribute="name", value="foo"),
        ],
    )
    expected = {
        "$and": [
            {"qty": {"$gte": 10}},
            {"qty": {"$lte": 20}},
            {"name": {"$eq": "foo"}},
        ]
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query_no_filter() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_one_attr() -> None:
    query = "What is the capital of France?"
    comp = Comparison(comparator=Comparator.IN, attribute="qty", value=[5, 15, 20])
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (
        query,
        {"filter": {"qty": {"$in": [5, 15, 20]}}},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_deep_nesting() -> None:
    query = "What is the capital of France?"
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="name", value="foo"),
            Operation(
                operator=Operator.OR,
                arguments=[
                    Comparison(comparator=Comparator.GT, attribute="qty", value=6),
                    Comparison(
                        comparator=Comparator.NIN,
                        attribute="tags",
                        value=["bar", "foo"],
                    ),
                ],
            ),
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
                    {"name": {"$eq": "foo"}},
                    {"$or": [{"qty": {"$gt": 6}}, {"tags": {"$nin": ["bar", "foo"]}}]},
                ]
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
