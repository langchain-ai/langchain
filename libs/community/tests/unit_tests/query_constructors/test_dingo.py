from typing import Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_community.query_constructors.dingo import DingoDBTranslator

DEFAULT_TRANSLATOR = DingoDBTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    expected = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
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
        {
            "search_params": {
                "langchain_expr": Comparison(
                    comparator=Comparator.LT, attribute="foo", value=["1", "2"]
                )
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
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
            "search_params": {
                "langchain_expr": Operation(
                    operator=Operator.AND,
                    arguments=[
                        Comparison(comparator=Comparator.LT, attribute="foo", value=2),
                        Comparison(
                            comparator=Comparator.EQ, attribute="bar", value="baz"
                        ),
                    ],
                )
            }
        },
    )

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
