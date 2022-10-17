import pytest

from langchain.formatting import formatter


def test_valid_formatting():
    template = "This is a {foo} test."
    output = formatter.format(template, foo="good")
    expected_output = "This is a good test."
    assert output == expected_output


def test_does_not_allow_args():
    template = "This is a {} test."
    with pytest.raises(ValueError):
        formatter.format(template, "good")


def test_does_not_allow_extra_kwargs():
    template = "This is a {foo} test."
    with pytest.raises(KeyError):
        formatter.format(template, foo="good", bar="oops")
