from typing import Dict, Tuple

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.vectara import VectaraTranslator

DEFAULT_TRANSLATOR = VectaraTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value="1")
    expected = "( doc.foo < '1' )"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=1),
        ],
    )
    expected = "( ( doc.foo < 2 ) and ( doc.bar = 'baz' ) and ( doc.abc < 1 ) )"
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
        limit=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = (query, {"filter": "( doc.foo < 1 )"})
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
        limit=None,
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=1),
        ],
    )
    structured_query = StructuredQuery(query=query, filter=op, limit=None)
    expected = (
        query,
        {"filter": "( ( doc.foo < 2 ) and ( doc.bar = 'baz' ) and ( doc.abc < 1 ) )"},
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
