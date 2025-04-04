import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)

from langchain_community.utils.mql import mql_to_filter


def test_mql_to_filter_eq() -> None:
    mql_filter = {"field": {"$eq": 10}}
    expected = Comparison(comparator=Comparator.EQ, attribute="field", value=10)
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_implicit_eq() -> None:
    mql_filter = {"field": 10}
    expected = Comparison(comparator=Comparator.EQ, attribute="field", value=10)
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_in() -> None:
    mql_filter = {"field": {"$in": [1, 2, 3]}}
    expected = Comparison(comparator=Comparator.IN, attribute="field", value=[1, 2, 3])
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_and() -> None:
    mql_filter = {"$and": [{"field1": {"$eq": 10}}, {"field2": {"$gt": 5}}]}
    expected = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="field1", value=10),
            Comparison(comparator=Comparator.GT, attribute="field2", value=5),
        ],
    )

    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_or() -> None:
    mql_filter = {"$or": [{"field1": {"$eq": 10}}, {"field2": {"$gt": 5}}]}
    expected = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="field1", value=10),
            Comparison(comparator=Comparator.GT, attribute="field2", value=5),
        ],
    )
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_not() -> None:
    mql_filter = {"$not": {"field": {"$eq": 10}}}
    expected = Operation(
        operator=Operator.NOT,
        arguments=[Comparison(comparator=Comparator.EQ, attribute="field", value=10)],
    )
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_nor() -> None:
    mql_filter = {"$nor": [{"field1": {"$eq": 10}}, {"field2": {"$gt": 5}}]}
    expected = Operation(
        operator=Operator.NOT,
        arguments=[
            Operation(
                operator=Operator.OR,
                arguments=[
                    Comparison(comparator=Comparator.EQ, attribute="field1", value=10),
                    Comparison(comparator=Comparator.GT, attribute="field2", value=5),
                ],
            )
        ],
    )
    assert mql_to_filter(mql_filter) == expected


def test_mql_to_filter_invalid_operator() -> None:
    mql_filter = {"field": {"$invalid": 10}}
    with pytest.raises(ValueError, match="Unsupported mql operator"):
        mql_to_filter(mql_filter)


def test_mql_to_filter_invalid_logical_operator() -> None:
    mql_filter = {"$and": {"field": {"$eq": 10}}}
    with pytest.raises(ValueError, match="must have a list"):
        mql_to_filter(mql_filter)


def test_mql_to_filter_invalid_logical_operand() -> None:
    mql_filter = {"$and": [10]}
    with pytest.raises(ValueError, match="must be dictionaries."):
        mql_to_filter(mql_filter)
