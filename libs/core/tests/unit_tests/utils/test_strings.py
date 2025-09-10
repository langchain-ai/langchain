"""Test string utilities."""

from langchain_core.utils.strings import (
    comma_list,
    sanitize_for_postgres,
    stringify_dict,
    stringify_value,
)


def test_sanitize_for_postgres() -> None:
    """Test sanitizing text for PostgreSQL compatibility."""
    # Test with NUL bytes
    text_with_nul = "Hello\x00world\x00test"
    expected = "Helloworldtest"
    assert sanitize_for_postgres(text_with_nul) == expected

    # Test with replacement character
    expected_with_replacement = "Hello world test"
    assert sanitize_for_postgres(text_with_nul, " ") == expected_with_replacement

    # Test with text without NUL bytes
    clean_text = "Hello world"
    assert sanitize_for_postgres(clean_text) == clean_text

    # Test empty string
    assert not sanitize_for_postgres("")

    # Test with multiple consecutive NUL bytes
    text_with_multiple_nuls = "Hello\x00\x00\x00world"
    assert sanitize_for_postgres(text_with_multiple_nuls) == "Helloworld"
    assert sanitize_for_postgres(text_with_multiple_nuls, "-") == "Hello---world"


def test_existing_string_functions() -> None:
    """Test existing string functions still work."""
    # Test comma_list
    assert comma_list([1, 2, 3]) == "1, 2, 3"
    assert comma_list(["a", "b", "c"]) == "a, b, c"

    # Test stringify_value
    assert stringify_value("hello") == "hello"
    assert stringify_value(42) == "42"

    # Test stringify_dict
    data = {"key": "value", "number": 123}
    result = stringify_dict(data)
    assert "key: value" in result
    assert "number: 123" in result
