"""Tests for array encoding patterns."""

from langchain_core.toon import encode


def test_encode_empty_array() -> None:
    """Test encoding an empty array."""
    result = encode([])
    assert result == "[0]:"


def test_encode_primitive_array_inline() -> None:
    """Test that arrays of primitives are encoded inline."""
    result = encode(["a", "b", "c"])
    assert result == "[3]: a,b,c"


def test_encode_integer_array_inline() -> None:
    """Test encoding array of integers inline."""
    result = encode([1, 2, 3, 4, 5])
    assert result == "[5]: 1,2,3,4,5"


def test_encode_mixed_primitive_array() -> None:
    """Test encoding array with mixed primitive types."""
    result = encode([1, "two", 3.0, True, None])
    assert result == "[5]: 1,two,3,true,null"


def test_encode_array_with_quoted_strings() -> None:
    """Test that strings needing quotes are quoted in arrays."""
    result = encode(["hello world", "simple", "key:value"])
    assert result == '[3]: "hello world",simple,"key:value"'


def test_encode_object_with_primitive_array() -> None:
    """Test encoding object containing a primitive array."""
    obj = {"tags": ["python", "ai", "ml"]}
    result = encode(obj)
    assert result == "tags[3]: python,ai,ml"


def test_encode_array_of_empty_arrays() -> None:
    """Test encoding array containing empty arrays."""
    result = encode([[], [], []])
    lines = result.split("\n")

    assert lines[0] == "[3]:"
    assert lines[1] == "  - [0]:"
    assert lines[2] == "  - [0]:"
    assert lines[3] == "  - [0]:"


def test_encode_array_of_primitive_arrays() -> None:
    """Test encoding array of arrays with primitive values."""
    result = encode([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lines = result.split("\n")

    assert lines[0] == "[3]:"
    assert lines[1] == "  - [3]: 1,2,3"
    assert lines[2] == "  - [3]: 4,5,6"
    assert lines[3] == "  - [3]: 7,8,9"


def test_encode_tabular_array_of_objects() -> None:
    """Test that arrays of objects with identical keys use tabular format."""
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    result = encode(users)
    lines = result.split("\n")

    assert lines[0] == "[3]{id,name}:"
    assert lines[1] == "  1,Alice"
    assert lines[2] == "  2,Bob"
    assert lines[3] == "  3,Charlie"


def test_encode_tabular_array_with_more_fields() -> None:
    """Test tabular format with more fields."""
    data = [
        {"id": 1, "name": "Alice", "age": 30, "active": True},
        {"id": 2, "name": "Bob", "age": 25, "active": False},
    ]
    result = encode(data)
    lines = result.split("\n")

    assert lines[0] == "[2]{id,name,age,active}:"
    assert lines[1] == "  1,Alice,30,true"
    assert lines[2] == "  2,Bob,25,false"


def test_encode_array_of_objects_non_tabular() -> None:
    """Test that arrays with non-uniform objects use list format."""
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "email": "bob@example.com"},  # Different keys
    ]
    result = encode(data)
    lines = result.split("\n")

    assert lines[0] == "[2]:"
    assert "- id: 1" in result
    assert "name: Alice" in result
    assert "- id: 2" in result
    assert "email: bob@example.com" in result


def test_encode_array_with_nested_objects() -> None:
    """Test encoding array with objects containing nested values."""
    data = [
        {"id": 1, "data": {"x": 10, "y": 20}},
        {"id": 2, "data": {"x": 30, "y": 40}},
    ]
    result = encode(data)
    lines = result.split("\n")

    # Non-tabular because values are not primitives
    assert lines[0] == "[2]:"
    assert "- id: 1" in result
    assert "data:" in result


def test_encode_mixed_array_as_list_items() -> None:
    """Test encoding mixed array with different value types."""
    mixed = [42, "text", {"key": "value"}, [1, 2, 3]]
    result = encode(mixed)
    lines = result.split("\n")

    assert lines[0] == "[4]:"
    assert "  - 42" in result
    assert "  - text" in result
    assert "  - key: value" in result
    assert "  - [3]: 1,2,3" in result


def test_encode_nested_array_structure() -> None:
    """Test deeply nested array structures."""
    data = {"matrix": [[1, 2], [3, 4], [5, 6]]}
    result = encode(data)
    lines = result.split("\n")

    assert lines[0] == "matrix[3]:"
    assert "  - [2]: 1,2" in result
    assert "  - [2]: 3,4" in result
    assert "  - [2]: 5,6" in result


def test_encode_array_with_null_values() -> None:
    """Test encoding arrays containing null values."""
    result = encode([1, None, 3, None, 5])
    assert result == "[5]: 1,null,3,null,5"


def test_encode_array_with_boolean_values() -> None:
    """Test encoding arrays with boolean values."""
    result = encode([True, False, True])
    assert result == "[3]: true,false,true"


def test_encode_object_with_array_key() -> None:
    """Test encoding object with array as value."""
    obj = {"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}
    result = encode(obj)

    assert "numbers[3]: 1,2,3" in result
    assert "letters[3]: a,b,c" in result


def test_encode_array_with_empty_strings() -> None:
    """Test encoding arrays with empty strings (should be quoted)."""
    result = encode(["a", "", "c"])
    assert result == '[3]: a,"",c'


def test_encode_single_element_array() -> None:
    """Test encoding array with single element."""
    result = encode([42])
    assert result == "[1]: 42"


def test_encode_array_of_single_object() -> None:
    """Test encoding array with single object (uses tabular format)."""
    result = encode([{"name": "Alice"}])
    lines = result.split("\n")

    # Single object with primitive value uses tabular format
    assert lines[0] == "[1]{name}:"
    assert lines[1] == "  Alice"
