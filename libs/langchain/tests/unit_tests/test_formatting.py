"""Test formatting functionality."""
import pytest

from langchain.utils import formatter


def test_valid_formatting() -> None:
    """Test formatting works as expected."""
    template = "This is a {foo} test."
    output = formatter.format(template, foo="good")
    expected_output = "This is a good test."
    assert output == expected_output


def test_does_not_allow_args() -> None:
    """Test formatting raises error when args are provided."""
    template = "This is a {} test."
    with pytest.raises(ValueError):
        formatter.format(template, "good")


def test_does_not_allow_extra_kwargs() -> None:
    """Test formatting does not allow extra keyword arguments."""
    template = "This is a {foo} test."
    with pytest.raises(KeyError):
        formatter.format(template, foo="good", bar="oops")
