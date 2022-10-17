"""Test helper functions for Cohere API."""

from langchain.llms.cohere import remove_stop_tokens


def test_remove_stop_tokens() -> None:
    """Test removing stop tokens when they occur."""
    text = "foo bar baz"
    output = remove_stop_tokens(text, ["moo", "baz"])
    assert output == "foo bar "


def test_remove_stop_tokens_none() -> None:
    """Test removing stop tokens when they do not occur."""
    text = "foo bar baz"
    output = remove_stop_tokens(text, ["moo"])
    assert output == "foo bar baz"
