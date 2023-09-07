"""Test LLM utility functions."""
from langchain.llms.utils import enforce_stop_tokens, remove_stop_words


def test_enforce_stop_tokens() -> None:
    """Test removing stop tokens when they occur."""
    text = "foo bar baz"
    output = enforce_stop_tokens(text, ["moo", "baz"])
    assert output == "foo bar "
    text = "foo bar baz"
    output = enforce_stop_tokens(text, ["moo", "baz", "bar"])
    assert output == "foo "
    text = "foo bar baz"
    output = enforce_stop_tokens(text, ["moo", "bar"])
    assert output == "foo "


def test_enforce_stop_tokens_none() -> None:
    """Test removing stop tokens when they do not occur."""
    text = "foo bar baz"
    output = enforce_stop_tokens(text, ["moo"])
    assert output == "foo bar baz"


def test_remove_stop_words() -> None:
    """Test removing stop tokens from the text"""
    text = "foo bar baz"
    output = remove_stop_words(text=text, stop=["baz"])
    assert "baz" not in output
