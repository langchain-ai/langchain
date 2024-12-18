from typing import Any, Dict, Tuple

import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.databricks_vector_search import (
    DatabricksVectorSearchTranslator,
)

DEFAULT_TRANSLATOR = DatabricksVectorSearchTranslator()


@pytest.mark.parametrize(
    "triplet",
    [
        (Comparator.EQ, 2, {"foo": 2}),
        (Comparator.GT, 2, {"foo >": 2}),
        (Comparator.GTE, 2, {"foo >=": 2}),
        (Comparator.LT, 2, {"foo <": 2}),
        (Comparator.LTE, 2, {"foo <=": 2}),
        (Comparator.IN, ["bar", "abc"], {"foo": ["bar", "abc"]}),
        (Comparator.LIKE, "bar", {"foo LIKE": "bar"}),
    ],
)
def test_visit_comparison(triplet: Tuple[Comparator, Any, str]) -> None:
    comparator, value, expected = triplet
    comp = Comparison(comparator=comparator, attribute="foo", value=value)
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation_and() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {"foo <": 2, "bar": "baz"}
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_operation_or() -> None:
    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = {"foo OR bar": [2, "baz"]}
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_operation_not() -> None:
    op = Operation(
        operator=Operator.NOT,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
        ],
    )
    expected = {"foo NOT": 2}
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_operation_not_that_raises_for_more_than_one_filter_condition() -> None:
    with pytest.raises(Exception) as exc_info:
        op = Operation(
            operator=Operator.NOT,
            arguments=[
                Comparison(comparator=Comparator.EQ, attribute="foo", value=2),
                Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            ],
        )
        DEFAULT_TRANSLATOR.visit_operation(op)
    assert (
        str(exc_info.value) == '"not" can have only one argument in '
        "Databricks vector search"
    )


def test_visit_structured_query_with_no_filter() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_with_one_arg_filter() -> None:
    query = "What is the capital of France?"
    comp = Comparison(comparator=Comparator.EQ, attribute="country", value="France")
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )

    expected = (query, {"filter": {"country": "France"}})

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_visit_structured_query_with_multiple_arg_filter_and_operator() -> None:
    query = "What is the capital of France in the years between 1888 and 1900?"

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="country", value="France"),
            Comparison(comparator=Comparator.GTE, attribute="year", value=1888),
            Comparison(comparator=Comparator.LTE, attribute="year", value=1900),
        ],
    )

    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )

    expected = (
        query,
        {"filter": {"country": "France", "year >=": 1888, "year <=": 1900}},
    )

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
