from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain.retrievers.self_query.pinecone import PineconeTranslator

DEFAULT_TRANSLATOR = PineconeTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    expected = {"foo": {"$lt": ["1", "2"]}}
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
    expected = {"$and": [{"foo": {"$lt": 2}}, {"bar": {"$eq": "baz"}}]}
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual
