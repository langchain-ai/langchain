from typing import Dict, Tuple

import pytest as pytest

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.retrievers.self_query.timescalevector import TimescaleVectorTranslator

DEFAULT_TRANSLATOR = TimescaleVectorTranslator()


@pytest.mark.requires("timescale_vector")
def test_visit_comparison() -> None:
    from timescale_vector import client

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = client.Predicates(("foo", "<", 1))
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


@pytest.mark.requires("timescale_vector")
def test_visit_operation() -> None:
    from timescale_vector import client

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    expected = client.Predicates(
        client.Predicates(("foo", "<", 2)),
        client.Predicates(("bar", "==", "baz")),
        client.Predicates(("abc", ">", 2.0)),
    )

    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


@pytest.mark.requires("timescale_vector")
def test_visit_structured_query() -> None:
    from timescale_vector import client

    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = (
        query,
        {"predicates": client.Predicates(("foo", "<", 1))},
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
            "predicates": client.Predicates(
                client.Predicates(("foo", "<", 2)),
                client.Predicates(("bar", "==", "baz")),
                client.Predicates(("abc", ">", 2.0)),
            )
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual
