"""Tests for object encoding and nested structures."""

from langchain_core.toon import encode


def test_encode_empty_object() -> None:
    """Test encoding an empty object."""
    result = encode({})
    assert result == ""


def test_encode_simple_object() -> None:
    """Test encoding a simple object with primitive values."""
    obj = {"name": "Alice", "age": 30, "active": True}
    result = encode(obj)

    assert "name: Alice" in result
    assert "age: 30" in result
    assert "active: true" in result


def test_encode_object_with_null_value() -> None:
    """Test encoding object with null value."""
    obj = {"name": "Bob", "email": None}
    result = encode(obj)

    assert "name: Bob" in result
    assert "email: null" in result


def test_encode_nested_object() -> None:
    """Test encoding nested objects."""
    obj = {"user": {"name": "Charlie", "id": 123}, "status": "active"}
    result = encode(obj)

    lines = result.split("\n")
    assert "user:" in lines[0]
    assert "name: Charlie" in result
    assert "id: 123" in result
    assert "status: active" in result


def test_encode_deeply_nested_object() -> None:
    """Test encoding deeply nested objects."""
    obj = {"a": {"b": {"c": {"d": "deep"}}}}
    result = encode(obj)

    assert "a:" in result
    assert "b:" in result
    assert "c:" in result
    assert "d: deep" in result


def test_encode_object_with_empty_nested_object() -> None:
    """Test encoding object containing an empty nested object."""
    obj = {"data": {}, "status": "ok"}
    result = encode(obj)

    assert "data:" in result
    assert "status: ok" in result


def test_encode_object_preserves_key_order() -> None:
    """Test that object key order is preserved."""
    obj = {"z": 1, "a": 2, "m": 3}
    result = encode(obj)
    lines = result.split("\n")

    # Check that keys appear in insertion order
    z_idx = next(i for i, line in enumerate(lines) if "z:" in line)
    a_idx = next(i for i, line in enumerate(lines) if "a:" in line)
    m_idx = next(i for i, line in enumerate(lines) if "m:" in line)

    assert z_idx < a_idx < m_idx


def test_encode_object_with_quoted_key() -> None:
    """Test encoding objects with keys that need quoting."""
    obj = {"first name": "Alice", "user-id": 123}
    result = encode(obj)

    assert '"first name": Alice' in result
    assert '"user-id": 123' in result


def test_encode_object_with_special_string_values() -> None:
    """Test encoding objects with strings that need escaping."""
    obj = {"message": "Hello\nWorld", "path": "C:\\Users\\test"}
    result = encode(obj)

    assert 'message: "Hello\\nWorld"' in result
    assert 'path: "C:\\\\Users\\\\test"' in result


def test_encode_object_indentation() -> None:
    """Test that nested objects use proper indentation."""
    obj = {"level1": {"level2": {"level3": "value"}}}
    result = encode(obj, indent=2)
    lines = result.split("\n")

    # Check indentation levels
    assert lines[0] == "level1:"
    assert lines[1].startswith("  level2:")
    assert lines[2].startswith("    level3: value")


def test_encode_object_custom_indentation() -> None:
    """Test encoding with custom indentation size."""
    obj = {"outer": {"inner": "value"}}
    result = encode(obj, indent=4)
    lines = result.split("\n")

    assert lines[0] == "outer:"
    assert lines[1].startswith("    inner: value")


def test_encode_object_with_numeric_keys() -> None:
    """Test encoding objects with numeric string keys."""
    obj = {"123": "numeric", "abc": "alpha"}
    result = encode(obj)

    # Numeric keys should be quoted
    assert '"123": numeric' in result
    assert "abc: alpha" in result


def test_encode_mixed_value_types() -> None:
    """Test encoding object with mixed value types."""
    obj = {
        "string": "text",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
    }
    result = encode(obj)

    assert "string: text" in result
    assert "number: 42" in result
    assert "float: 3.14" in result
    assert "bool: true" in result
    assert "null: null" in result
