from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain.retrievers.self_query.deeplake import DeepLakeTranslator

DEFAULT_TRANSLATOR = DeepLakeTranslator()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=["1", "2"])
    expected = "(metadata['foo'] < 1 or metadata['foo'] < 2)"
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LT, attribute="abc", value=["1", "2"]),
        ],
    )
    expected = (
        "(metadata['foo'] < 2 and metadata['bar'] == 'baz' "
        "and (metadata['abc'] < 1 or metadata['abc'] < 2))"
    )
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual
