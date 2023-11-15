from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.milvus import MilvusTranslator

DEFAULT_TRANSLATOR = MilvusTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=4)
    expected = "( foo < 4 )"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)

    assert expected == actual


def test_visit_operation() -> None:
    # Non-Unary operator

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value="4"),
        ],
    )

    expected = '(( foo < 2 ) and ( bar == "baz" ) ' 'and ( abc < "4" ))'
    actual = DEFAULT_TRANSLATOR.visit_operation(op)

    assert expected == actual

    # Unary operator: normal execution
    op = Operation(
        operator=Operator.NOT,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
        ],
    )

    expected = "not(( foo < 2 ))"
    actual = DEFAULT_TRANSLATOR.visit_operation(op)

    assert expected == actual

    # Unary operator: error
    op = Operation(
        operator=Operator.NOT,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value="4"),
        ],
    )

    try:
        DEFAULT_TRANSLATOR.visit_operation(op)
    except ValueError as e:
        assert str(e) == '"not" can have only one argument in Milvus'
    else:
        assert False, "Expected exception not raised"  # No exception -> test failed


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=454)
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )

    expected = (
        query,
        {"expr": "( foo < 454 )"},
    )

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=50),
        ],
    )

    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )

    expected = (
        query,
        {"expr": "(( foo < 2 ) " 'and ( bar == "baz" ) ' "and ( abc < 50 ))"},
    )

    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
