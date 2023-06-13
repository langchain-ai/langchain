import pytest

from typing import Tuple, Any
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)
from langchain.retrievers.self_query.myscale import MyScaleTranslator

DEFAULT_TRANSLATOR = MyScaleTranslator()


@pytest.mark.parametrize(
    "triplet",
    [
        (Comparator.LT, 2, "foo < 2"),
        (Comparator.LTE, 2, "foo <= 2"),
        (Comparator.GT, 2, "foo > 2"),
        (Comparator.GTE, 2, "foo >= 2"),
        (Comparator.CONTAIN, 2, "has(foo, 2)"),
        (Comparator.LIKE, "bar", "foo ILKE 'bar'"),
    ],
)
def test_visit_comparison(triplet: Tuple[Comparator, Any, str]) -> None:
    comparator, value, expected = triplet
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=2)
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


# def test_visit_operation() -> None:
#     op = Operation(
#         operator=Operator.AND,
#         arguments=[
#             Comparison(comparator=Comparator.LT, attribute="foo", value=2),
#             Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
#         ],
#     )
#     expected = {"$and": [{"foo": {"$lt": 2}}, {"bar": {"$eq": "baz"}}]}
#     actual = DEFAULT_TRANSLATOR.visit_operation(op)
#     assert expected == actual
