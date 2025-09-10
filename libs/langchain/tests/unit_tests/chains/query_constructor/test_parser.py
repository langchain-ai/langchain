"""Test LLM-generated structured query parsing."""

from typing import Any, Optional, cast

import lark
import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
)

from langchain.chains.query_constructor.parser import get_parser

DEFAULT_PARSER = get_parser()


@pytest.mark.parametrize("x", ["", "foo", 'foo("bar", "baz")'])
def test_parse_invalid_grammar(x: str) -> None:
    with pytest.raises((ValueError, lark.exceptions.UnexpectedToken)):
        DEFAULT_PARSER.parse(x)


def test_parse_comparison() -> None:
    comp = 'gte("foo", 2)'
    expected = Comparison(comparator=Comparator.GTE, attribute="foo", value=2)
    for text in (
        comp,
        comp.replace('"', "'"),
        comp.replace(" ", ""),
        comp.replace(" ", "  "),
        comp.replace("(", " ("),
        comp.replace(",", ", "),
        comp.replace("2", "2.0"),
    ):
        actual = DEFAULT_PARSER.parse(text)
        assert expected == actual


def test_parse_operation() -> None:
    op = 'and(eq("foo", "bar"), lt("baz", 1995.25))'
    eq = Comparison(comparator=Comparator.EQ, attribute="foo", value="bar")
    lt = Comparison(comparator=Comparator.LT, attribute="baz", value=1995.25)
    expected = Operation(operator=Operator.AND, arguments=[eq, lt])
    for text in (
        op,
        op.replace('"', "'"),
        op.replace(" ", ""),
        op.replace(" ", "  "),
        op.replace("(", " ("),
        op.replace(",", ", "),
        op.replace("25", "250"),
    ):
        actual = DEFAULT_PARSER.parse(text)
        assert expected == actual


def test_parse_nested_operation() -> None:
    op = 'and(or(eq("a", "b"), eq("a", "c"), eq("a", "d")), not(eq("z", "foo")))'
    eq1 = Comparison(comparator=Comparator.EQ, attribute="a", value="b")
    eq2 = Comparison(comparator=Comparator.EQ, attribute="a", value="c")
    eq3 = Comparison(comparator=Comparator.EQ, attribute="a", value="d")
    eq4 = Comparison(comparator=Comparator.EQ, attribute="z", value="foo")
    _not = Operation(operator=Operator.NOT, arguments=[eq4])
    _or = Operation(operator=Operator.OR, arguments=[eq1, eq2, eq3])
    expected = Operation(operator=Operator.AND, arguments=[_or, _not])
    actual = DEFAULT_PARSER.parse(op)
    assert expected == actual


def test_parse_disallowed_comparator() -> None:
    parser = get_parser(allowed_comparators=[Comparator.EQ])
    with pytest.raises(ValueError, match="Received disallowed comparator gt."):
        parser.parse('gt("a", 2)')


def test_parse_disallowed_operator() -> None:
    parser = get_parser(allowed_operators=[Operator.AND])
    with pytest.raises(ValueError, match="Received disallowed operator not."):
        parser.parse('not(gt("a", 2))')


def _test_parse_value(x: Any) -> None:
    parsed = cast("Comparison", (DEFAULT_PARSER.parse(f'eq("x", {x})')))
    actual = parsed.value
    assert actual == x


@pytest.mark.parametrize("x", [-1, 0, 1_000_000])
def test_parse_int_value(x: int) -> None:
    _test_parse_value(x)


@pytest.mark.parametrize("x", [-1.001, 0.00000002, 1_234_567.6543210])
def test_parse_float_value(x: float) -> None:
    _test_parse_value(x)


@pytest.mark.parametrize("x", [[], [1, "b", "true"]])
def test_parse_list_value(x: list) -> None:
    _test_parse_value(x)


@pytest.mark.parametrize("x", ['""', '" "', '"foo"', "'foo'"])
def test_parse_string_value(x: str) -> None:
    parsed = cast("Comparison", DEFAULT_PARSER.parse(f'eq("x", {x})'))
    actual = parsed.value
    assert actual == x[1:-1]


@pytest.mark.parametrize("x", ["true", "True", "TRUE", "false", "False", "FALSE"])
def test_parse_bool_value(x: str) -> None:
    parsed = cast("Comparison", DEFAULT_PARSER.parse(f'eq("x", {x})'))
    actual = parsed.value
    expected = x.lower() == "true"
    assert actual == expected


@pytest.mark.parametrize("op", ["and", "or"])
@pytest.mark.parametrize("arg", ['eq("foo", 2)', 'and(eq("foo", 2), lte("bar", 1.1))'])
def test_parser_unpack_single_arg_operation(op: str, arg: str) -> None:
    expected = DEFAULT_PARSER.parse(arg)
    actual = DEFAULT_PARSER.parse(f"{op}({arg})")
    assert expected == actual


@pytest.mark.parametrize("x", ['"2022-10-20"', "'2022-10-20'", "2022-10-20"])
def test_parse_date_value(x: str) -> None:
    parsed = cast("Comparison", DEFAULT_PARSER.parse(f'eq("x", {x})'))
    actual = parsed.value["date"]
    assert actual == x.strip("'\"")


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (
            '"2021-01-01T00:00:00"',
            {"datetime": "2021-01-01T00:00:00", "type": "datetime"},
        ),
        (
            '"2021-12-31T23:59:59Z"',
            {"datetime": "2021-12-31T23:59:59Z", "type": "datetime"},
        ),
    ],
)
def test_parse_datetime_value(x: str, expected: Optional[dict[str, str]]) -> None:
    """Test parsing of datetime values with ISO 8601 format."""
    parsed = cast("Comparison", DEFAULT_PARSER.parse(f'eq("publishedAt", {x})'))
    actual = parsed.value
    assert actual == expected, f"Expected {expected}, got {actual}"
