"""Test LLM utility functions."""
from langchain.llms.utils import enforce_stop_tokens


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
    text = "foo.bar.baz"
    output = enforce_stop_tokens(text, ["."])
    assert output == "foo"
    text = "My favorite number is 7"
    output = enforce_stop_tokens(text, ["\d"])
    assert output == text
    text = "Hello\nWorld"
    output = enforce_stop_tokens(text, ["\n"])
    assert output == "Hello"


def test_enforce_stop_tokens_none() -> None:
    """Test removing stop tokens when they do not occur."""
    text = "foo bar baz"
    output = enforce_stop_tokens(text, ["moo"])
    assert output == "foo bar baz"
