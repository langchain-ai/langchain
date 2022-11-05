"""Test input manipulating logic."""

import sys
from io import StringIO

from langchain.input import ChainedInput, get_color_mapping


def test_chained_input_not_verbose() -> None:
    """Test chained input logic."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    chained_input = ChainedInput("foo")
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    assert output == ""
    assert chained_input.input == "foo"

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    chained_input.add("bar")
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    assert output == ""
    assert chained_input.input == "foobar"


def test_chained_input_verbose() -> None:
    """Test chained input logic, making sure verbose doesn't mess it up."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    chained_input = ChainedInput("foo", verbose=True)
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    assert output == "foo"
    assert chained_input.input == "foo"

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    chained_input.add("bar")
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    assert output == "bar"
    assert chained_input.input == "foobar"

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    chained_input.add("baz", color="blue")
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    assert output == "\x1b[104mbaz\x1b[0m"
    assert chained_input.input == "foobarbaz"


def test_get_color_mapping() -> None:
    """Test getting of color mapping."""
    # Test on few inputs.
    items = ["foo", "bar"]
    output = get_color_mapping(items)
    expected_output = {"foo": "blue", "bar": "yellow"}
    assert output == expected_output

    # Test on a lot of inputs.
    items = [f"foo-{i}" for i in range(20)]
    output = get_color_mapping(items)
    assert len(output) == 20


def test_get_color_mapping_excluded_colors() -> None:
    """Test getting of color mapping with excluded colors."""
    items = ["foo", "bar"]
    output = get_color_mapping(items, excluded_colors=["blue"])
    expected_output = {"foo": "yellow", "bar": "red"}
    assert output == expected_output
