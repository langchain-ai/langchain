import pytest

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain.retrievers.self_query.qdrant import QdrantTranslator

DEFAULT_TRANSLATOR = QdrantTranslator("payload")


@pytest.mark.requires("qdrant_client")
def test_visit_comparison() -> None:
    from qdrant_client.http import models as rest

    comp = Comparison(comparator=Comparator.GTE, attribute="foo", value=2.5)
    expected = rest.FieldCondition(key="payload.foo", range=rest.Range(gte=2.5))
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


@pytest.mark.requires("qdrant_client")
def test_visit_operation() -> None:
    from qdrant_client.http import models as rest

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
        ],
    )
    expected = rest.Filter(
        must=[
            rest.FieldCondition(key="payload.foo", range=rest.Range(lt=2)),
            rest.FieldCondition(key="payload.bar", match=rest.MatchValue(value="baz")),
        ]
    )
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual
