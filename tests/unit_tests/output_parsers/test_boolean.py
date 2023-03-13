from typing import List

import pytest

from langchain.output_parsers.boolean import BooleanOutputParser

GOOD_EXAMPLES = [
    ("0", False, ["1"], ["0"]),
    ("1", True, ["1"], ["0"]),
    ("\n1\n", True, ["1"], ["0"]),
    ("The answer is: \n1\n", True, ["1"], ["0"]),
    ("The answer is: 0", False, ["1"], ["0"]),
    ("1", False, ["0"], ["1"]),
    ("0", True, ["0"], ["1"]),
    ("X", True, ["x", "X"], ["O", "o"]),
]


@pytest.mark.parametrize(
    "input_string,expected,true_values,false_values", GOOD_EXAMPLES
)
def test_boolean_output_parsing(
    input_string: str, expected: str, true_values: List[str], false_values: List[str]
) -> None:
    """Test booleans are parsed as expected."""
    output_parser = BooleanOutputParser(
        true_values=true_values, false_values=false_values
    )
    output = output_parser.parse(input_string)
    assert output == expected


BAD_VALUES = [
    ("01", ["1"], ["0"]),
    ("", ["1"], ["0"]),
    ("a", ["0"], ["1"]),
    ("2", ["1"], ["0"]),
]


@pytest.mark.parametrize("input_string,true_values,false_values", BAD_VALUES)
def test_boolean_output_parsing_error(
    input_string: str, true_values: List[str], false_values: List[str]
) -> None:
    """Test errors when parsing."""
    output_parser = BooleanOutputParser(
        true_values=true_values, false_values=false_values
    )
    with pytest.raises(ValueError):
        output_parser.parse(input_string)


def test_boolean_output_parsing_init_error() -> None:
    """Test that init errors when bad values are passed to boolean output parser."""
    with pytest.raises(ValueError):
        BooleanOutputParser(true_values=["0", "1"], false_values=["0", "1"])
