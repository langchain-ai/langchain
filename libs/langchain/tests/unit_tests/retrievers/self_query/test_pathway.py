from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.pathway import PathwayTranslator

DEFAULT_TRANSLATOR = PathwayTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="foo", value="1")
    expected = "foo == `1`"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_integer() -> None:
    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=1)
    expected = "foo >= `1`"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_number() -> None:
    comp = Comparison(comparator=Comparator.GT, attribute="foo", value=1.4)
    expected = "foo > `1.4`"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_boolean() -> None:
    comp = Comparison(comparator=Comparator.NE, attribute="foo", value=False)
    expected = "foo != `false`"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_contain() -> None:
    comp = Comparison(comparator=Comparator.CONTAIN, attribute="foo", value="abc")
    expected = "contains(foo, `abc`)"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_comparison_datetime() -> None:
    comp = Comparison(
        comparator=Comparator.LTE,
        attribute="foo",
        value={"type": "date", "date": "2023-09-13"},
    )
    expected = "foo <= `1694556000`"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="foo", value="hello"),
            Comparison(
                comparator=Comparator.GTE,
                attribute="bar",
                value=1,
            ),
            Comparison(comparator=Comparator.LTE, attribute="abc", value=1.4),
        ],
    )
    expected = "foo == `hello` && bar >= `1` && abc <= `1.4`"
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
        {"metadata_filter": "foo == `1`"},
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
        {"metadata_filter": "foo == `2` && bar == `baz`"},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
