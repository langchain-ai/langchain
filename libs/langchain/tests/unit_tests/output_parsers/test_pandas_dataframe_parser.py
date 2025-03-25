"""Test PandasDataframeParser"""

from typing import Any, Dict

import pandas as pd
from langchain_core.exceptions import OutputParserException

from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser

df = pd.DataFrame(
    {
        "chicken": [1, 2, 3, 4],
        "veggies": [5, 4, 3, 2],
        "steak": [9, 8, 7, 6],
    }
)

parser = PandasDataFrameOutputParser(dataframe=df)


# Test Invalid Column
def test_pandas_output_parser_col_no_array() -> None:
    try:
        parser.parse("column:num_legs")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        assert True


# Test Column with invalid array (above DataFrame max index)
def test_pandas_output_parser_col_oob() -> None:
    try:
        parser.parse("row:10")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        assert True


# Test Column with array [x]
def test_pandas_output_parser_col_first_elem() -> None:
    expected_output = {"chicken": 1}
    actual_output = parser.parse("column:chicken[0]")
    assert actual_output == expected_output


# Test Column with array [x,y,z]
def test_pandas_output_parser_col_multi_elem() -> None:
    expected_output = {"chicken": pd.Series([1, 2], name="chicken", dtype="int64")}
    actual_output = parser.parse("column:chicken[0, 1]")
    for key in actual_output.keys():
        assert expected_output["chicken"].equals(actual_output[key])


# Test Row with invalid row entry
def test_pandas_output_parser_row_no_array() -> None:
    try:
        parser.parse("row:5")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        assert True


# Test Row with valid row entry
def test_pandas_output_parser_row_first() -> None:
    expected_output = {"1": pd.Series({"chicken": 2, "veggies": 4, "steak": 8})}
    actual_output = parser.parse("row:1")
    assert actual_output["1"].equals(expected_output["1"])


# Test Row with invalid col entry
def test_pandas_output_parser_row_no_column() -> None:
    try:
        parser.parse("row:1[num_legs]")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        assert True


# Test Row with valid col entry
def test_pandas_output_parser_row_col_1() -> None:
    expected_output = {"1": 2}
    actual_output = parser.parse("row:1[chicken]")
    assert actual_output == expected_output


def test_pandas_output_parser_special_ops() -> None:
    actual_output = [
        {"mean": 3.0},
        {"median": 3.0},
        {"min": 2},
        {"max": 4},
        {"var": 1.0},
        {"std": 1.0},
        {"count": 3},
        {"quantile": 3.0},
    ]

    expected_output = [
        parser.parse("mean:chicken[1..3]"),
        parser.parse("median:chicken[1..3]"),
        parser.parse("min:chicken[1..3]"),
        parser.parse("max:chicken[1..3]"),
        parser.parse("var:chicken[1..3]"),
        parser.parse("std:chicken[1..3]"),
        parser.parse("count:chicken[1..3]"),
        parser.parse("quantile:chicken[1..3]"),
    ]

    assert actual_output == expected_output


def test_pandas_output_parser_invalid_special_op() -> None:
    try:
        parser.parse("riemann_sum:chicken")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        assert True


def test_pandas_output_parser_output_type() -> None:
    """Test the output type of the pandas dataframe output parser is a pandas dataframe."""  # noqa: E501
    assert parser.OutputType is Dict[str, Any]
