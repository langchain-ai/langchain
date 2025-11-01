"""Tests for primitive value encoding and string escaping."""

from langchain_core.toon import encode
from langchain_core.toon.formatters import (
    encode_key,
    encode_primitive,
    encode_string_literal,
    escape_string,
    is_safe_unquoted,
)


def test_encode_none() -> None:
    """Test encoding None as null."""
    assert encode_primitive(None) == "null"
    assert encode(None) == "null"


def test_encode_bool_true() -> None:
    """Test encoding True as true."""
    assert encode_primitive(True) == "true"
    assert encode(True) == "true"


def test_encode_bool_false() -> None:
    """Test encoding False as false."""
    assert encode_primitive(False) == "false"
    assert encode(False) == "false"


def test_encode_integer() -> None:
    """Test encoding integers."""
    assert encode_primitive(0) == "0"
    assert encode_primitive(42) == "42"
    assert encode_primitive(-123) == "-123"
    assert encode(999) == "999"


def test_encode_float() -> None:
    """Test encoding floating point numbers."""
    assert encode_primitive(3.14) == "3.14"
    assert encode_primitive(0.5) == "0.5"
    assert encode_primitive(-2.718) == "-2.718"


def test_encode_float_removes_trailing_zeros() -> None:
    """Test that trailing zeros are removed from floats."""
    assert encode_primitive(1.0) == "1"
    assert encode_primitive(2.5000) == "2.5"
    assert encode_primitive(0.0) == "0"


def test_encode_simple_string() -> None:
    """Test encoding simple unquoted strings."""
    assert encode_primitive("hello") == "hello"
    assert encode_primitive("world123") == "world123"
    assert encode_primitive("snake_case") == "snake_case"


def test_encode_string_with_spaces_needs_quotes() -> None:
    """Test that strings with spaces are quoted."""
    assert encode_primitive("hello world") == '"hello world"'
    assert encode_primitive("  leading") == '"  leading"'
    assert encode_primitive("trailing  ") == '"trailing  "'


def test_encode_string_with_delimiter_needs_quotes() -> None:
    """Test that strings containing delimiters are quoted."""
    assert encode_primitive("a,b", ",") == '"a,b"'
    assert encode_primitive("a|b", "|") == '"a|b"'
    # Tab gets escaped when quoted
    assert encode_primitive("a\tb", "\t") == '"a\\tb"'


def test_encode_string_with_colon_needs_quotes() -> None:
    """Test that strings with colons are quoted."""
    assert encode_primitive("key:value") == '"key:value"'
    assert encode_primitive("http://example.com") == '"http://example.com"'


def test_encode_string_with_brackets_needs_quotes() -> None:
    """Test that strings with structural characters are quoted."""
    assert encode_primitive("array[0]") == '"array[0]"'
    assert encode_primitive("{key}") == '"{key}"'


def test_encode_reserved_literals_are_quoted() -> None:
    """Test that reserved literals are quoted to avoid ambiguity."""
    assert encode_primitive("true") == '"true"'
    assert encode_primitive("false") == '"false"'
    assert encode_primitive("null") == '"null"'


def test_encode_numeric_like_strings_are_quoted() -> None:
    """Test that numeric-like strings are quoted."""
    assert encode_primitive("123") == '"123"'
    assert encode_primitive("-456") == '"-456"'
    assert encode_primitive("3.14") == '"3.14"'
    assert encode_primitive("1e10") == '"1e10"'


def test_encode_string_starting_with_dash_needs_quotes() -> None:
    """Test that strings starting with dash are quoted."""
    assert encode_primitive("-item") == '"-item"'


def test_escape_string_backslash() -> None:
    """Test escaping backslashes."""
    assert escape_string("path\\to\\file") == "path\\\\to\\\\file"


def test_escape_string_quotes() -> None:
    """Test escaping double quotes."""
    assert escape_string('He said "hello"') == 'He said \\"hello\\"'


def test_escape_string_newline() -> None:
    """Test escaping newlines."""
    assert escape_string("line1\nline2") == "line1\\nline2"


def test_escape_string_carriage_return() -> None:
    """Test escaping carriage returns."""
    assert escape_string("line1\rline2") == "line1\\rline2"


def test_escape_string_tab() -> None:
    """Test escaping tabs."""
    assert escape_string("col1\tcol2") == "col1\\tcol2"


def test_encode_string_literal_with_escapes() -> None:
    """Test that strings with special characters are escaped and quoted."""
    assert encode_string_literal("line1\nline2") == '"line1\\nline2"'
    assert encode_string_literal('say "hi"') == '"say \\"hi\\""'
    assert encode_string_literal("path\\file") == '"path\\\\file"'


def test_is_safe_unquoted_alphanumeric() -> None:
    """Test that simple alphanumeric strings are safe unquoted."""
    assert is_safe_unquoted("hello")
    assert is_safe_unquoted("world123")
    assert is_safe_unquoted("snake_case")
    assert is_safe_unquoted("camelCase")


def test_is_safe_unquoted_empty_string() -> None:
    """Test that empty strings must be quoted."""
    assert not is_safe_unquoted("")


def test_is_safe_unquoted_whitespace() -> None:
    """Test that strings with whitespace must be quoted."""
    assert not is_safe_unquoted("hello world")
    assert not is_safe_unquoted(" leading")
    assert not is_safe_unquoted("trailing ")


def test_encode_key_simple() -> None:
    """Test encoding simple object keys."""
    assert encode_key("name") == "name"
    assert encode_key("user_id") == "user_id"
    assert encode_key("camelCase") == "camelCase"


def test_encode_key_with_dots() -> None:
    """Test encoding keys with dots."""
    assert encode_key("user.name") == "user.name"
    assert encode_key("a.b.c") == "a.b.c"


def test_encode_key_with_spaces_needs_quotes() -> None:
    """Test that keys with spaces are quoted."""
    assert encode_key("first name") == '"first name"'
    assert encode_key("user data") == '"user data"'


def test_encode_key_starting_with_number_needs_quotes() -> None:
    """Test that keys starting with numbers are quoted."""
    assert encode_key("123key") == '"123key"'
    assert encode_key("1st_place") == '"1st_place"'


def test_encode_key_with_special_chars_needs_quotes() -> None:
    """Test that keys with special characters are quoted."""
    assert encode_key("key:value") == '"key:value"'
    assert encode_key("key[0]") == '"key[0]"'
    assert encode_key("key-name") == '"key-name"'
