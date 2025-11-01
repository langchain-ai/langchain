"""Tests for encoding options (indent, delimiter, length_marker)."""

from langchain_core.toon import encode


def test_encode_with_default_indent() -> None:
    """Test default indentation is 2 spaces."""
    obj = {"outer": {"inner": "value"}}
    result = encode(obj)
    lines = result.split("\n")

    assert lines[0] == "outer:"
    assert lines[1] == "  inner: value"


def test_encode_with_custom_indent_4() -> None:
    """Test custom indentation of 4 spaces."""
    obj = {"outer": {"inner": "value"}}
    result = encode(obj, indent=4)
    lines = result.split("\n")

    assert lines[0] == "outer:"
    assert lines[1] == "    inner: value"


def test_encode_with_custom_indent_0() -> None:
    """Test no indentation (indent=0)."""
    obj = {"outer": {"inner": "value"}}
    result = encode(obj, indent=0)
    lines = result.split("\n")

    assert lines[0] == "outer:"
    assert lines[1] == "inner: value"


def test_encode_with_indent_nested_levels() -> None:
    """Test indentation across multiple nesting levels."""
    obj = {"a": {"b": {"c": "deep"}}}
    result = encode(obj, indent=3)
    lines = result.split("\n")

    assert lines[0] == "a:"
    assert lines[1] == "   b:"
    assert lines[2] == "      c: deep"


def test_encode_with_comma_delimiter() -> None:
    """Test default comma delimiter for arrays."""
    result = encode([1, 2, 3], delimiter=",")
    assert result == "[3]: 1,2,3"


def test_encode_with_pipe_delimiter() -> None:
    """Test pipe delimiter for arrays."""
    result = encode([1, 2, 3], delimiter="|")
    assert result == "[3|]: 1|2|3"


def test_encode_with_tab_delimiter() -> None:
    """Test tab delimiter for arrays."""
    result = encode([1, 2, 3], delimiter="\t")
    # Tab delimiter header includes the tab marker
    assert result.startswith("[3\t]:")
    assert "\t" in result


def test_encode_tabular_with_pipe_delimiter() -> None:
    """Test tabular format with pipe delimiter."""
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    result = encode(users, delimiter="|")
    lines = result.split("\n")

    assert lines[0] == "[2|]{id|name}:"
    assert lines[1] == "  1|Alice"
    assert lines[2] == "  2|Bob"


def test_encode_with_length_marker_false() -> None:
    """Test default length marker (no #)."""
    result = encode([1, 2, 3], length_marker=False)
    assert result == "[3]: 1,2,3"


def test_encode_with_length_marker_hash() -> None:
    """Test length marker with #."""
    result = encode([1, 2, 3], length_marker="#")
    assert result == "[#3]: 1,2,3"


def test_encode_tabular_with_length_marker() -> None:
    """Test tabular format with length marker."""
    users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    result = encode(users, length_marker="#")
    lines = result.split("\n")

    assert lines[0] == "[#2]{id,name}:"
    assert lines[1] == "  1,Alice"
    assert lines[2] == "  2,Bob"


def test_encode_empty_array_with_length_marker() -> None:
    """Test empty array with length marker."""
    result = encode([], length_marker="#")
    assert result == "[#0]:"


def test_encode_nested_array_with_length_marker() -> None:
    """Test nested arrays with length marker."""
    data = [[1, 2], [3, 4]]
    result = encode(data, length_marker="#")
    lines = result.split("\n")

    assert lines[0] == "[#2]:"
    assert "  - [#2]: 1,2" in result
    assert "  - [#2]: 3,4" in result


def test_encode_combined_options() -> None:
    """Test combining multiple options together."""
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    result = encode(users, indent=4, delimiter="|", length_marker="#")
    lines = result.split("\n")

    assert lines[0] == "[#2|]{id|name}:"
    assert lines[1] == "    1|Alice"
    assert lines[2] == "    2|Bob"


def test_encode_object_array_with_all_options() -> None:
    """Test object with array using all custom options (tabular format)."""
    obj = {"users": [{"id": 1}, {"id": 2}]}
    result = encode(obj, indent=3, delimiter="|", length_marker="#")

    # Uniform objects use tabular format
    assert "users[#2|]{id}:" in result
    assert "   1" in result
    assert "   2" in result


def test_encode_string_with_pipe_delimiter() -> None:
    """Test that strings containing pipe are quoted when pipe is delimiter."""
    data = ["a|b", "c", "d"]
    result = encode(data, delimiter="|")
    assert result == '[3|]: "a|b"|c|d'


def test_encode_string_with_comma_safe_for_pipe() -> None:
    """Test that strings with comma are safe when using pipe delimiter."""
    data = ["a,b", "c", "d"]
    result = encode(data, delimiter="|")
    # Comma doesn't need quoting with pipe delimiter
    assert result == "[3|]: a,b|c|d"
