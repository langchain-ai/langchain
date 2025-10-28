"""Tests for edge cases and special scenarios."""

from datetime import date, datetime

from langchain_core.toon import encode
from langchain_core.toon.normalize import normalize_value


def test_encode_unicode_strings() -> None:
    """Test encoding unicode characters (quoted due to spaces)."""
    result = encode({"text": "Hello 世界 🌍"})
    # Strings with spaces get quoted
    assert 'text: "Hello 世界 🌍"' in result


def test_encode_unicode_in_array() -> None:
    """Test encoding unicode in arrays."""
    result = encode(["α", "β", "γ"])
    assert result == "[3]: α,β,γ"


def test_encode_very_long_string() -> None:
    """Test encoding very long strings."""
    long_str = "a" * 1000
    result = encode({"data": long_str})
    assert f"data: {'a' * 1000}" in result


def test_encode_large_array() -> None:
    """Test encoding large arrays."""
    large_array = list(range(100))
    result = encode(large_array)
    assert result.startswith("[100]:")
    assert "99" in result


def test_encode_deeply_nested_structure() -> None:
    """Test encoding very deeply nested structures."""
    nested = {"a": {"b": {"c": {"d": {"e": {"f": "deep"}}}}}}
    result = encode(nested)
    assert "f: deep" in result


def test_encode_datetime_object() -> None:
    """Test encoding datetime objects as ISO strings (quoted due to colons)."""
    dt = datetime(2024, 1, 15, 12, 30, 45)
    result = encode({"timestamp": dt})
    # ISO strings contain colons so they get quoted
    assert 'timestamp: "2024-01-15T12:30:45"' in result


def test_encode_date_object() -> None:
    """Test encoding date objects as ISO strings."""
    d = date(2024, 1, 15)
    result = encode({"date": d})
    assert "date: 2024-01-15" in result


def test_encode_set() -> None:
    """Test encoding sets as arrays."""
    result = encode({"tags": {"python", "ai", "ml"}})
    # Sets become arrays, but order is not guaranteed
    assert "tags[3]:" in result


def test_encode_frozenset() -> None:
    """Test encoding frozensets as arrays."""
    result = encode({"items": frozenset([1, 2, 3])})
    assert "items[3]:" in result


def test_normalize_nan_to_null() -> None:
    """Test that NaN is normalized to null."""
    result = normalize_value(float("nan"))
    assert result is None


def test_normalize_infinity_to_null() -> None:
    """Test that infinity is normalized to null."""
    result = normalize_value(float("inf"))
    assert result is None

    result = normalize_value(float("-inf"))
    assert result is None


def test_normalize_negative_zero() -> None:
    """Test that -0.0 is normalized to 0."""
    result = normalize_value(-0.0)
    assert result == 0


def test_encode_mixed_nested_arrays() -> None:
    """Test encoding complex nested array structures."""
    data = [
        [1, 2, 3],
        ["a", "b", "c"],
        [True, False, True],
    ]
    result = encode(data)
    lines = result.split("\n")

    assert lines[0] == "[3]:"
    assert "  - [3]: 1,2,3" in result
    assert "  - [3]: a,b,c" in result
    assert "  - [3]: true,false,true" in result


def test_encode_array_with_objects_and_primitives() -> None:
    """Test encoding array with mixed objects and primitives."""
    data = [{"key": "value"}, 42, "text"]
    result = encode(data)

    assert "[3]:" in result
    assert "- key: value" in result
    assert "- 42" in result
    assert "- text" in result


def test_encode_empty_nested_structures() -> None:
    """Test encoding structures with empty nested values."""
    data = {"empty_obj": {}, "empty_array": [], "value": 42}
    result = encode(data)

    assert "empty_obj:" in result
    assert "empty_array[0]:" in result
    assert "value: 42" in result


def test_encode_special_float_values() -> None:
    """Test encoding special float values."""
    data = {"pi": 3.14159, "e": 2.71828, "zero": 0.0, "negative": -1.5}
    result = encode(data)

    assert "pi: 3.14159" in result
    assert "e: 2.71828" in result
    assert "zero: 0" in result
    assert "negative: -1.5" in result


def test_encode_string_with_all_escapes() -> None:
    """Test encoding string with all escape sequences."""
    text = 'Line1\nLine2\rTab\there"Quote"Back\\slash'
    result = encode({"text": text})
    assert "text:" in result
    assert "\\n" in result
    assert "\\r" in result
    assert "\\t" in result
    assert '\\"' in result
    assert "\\\\" in result


def test_encode_keys_with_dots() -> None:
    """Test encoding keys containing dots."""
    obj = {"user.name": "Alice", "user.id": 123}
    result = encode(obj)

    assert "user.name: Alice" in result
    assert "user.id: 123" in result


def test_encode_numeric_string_keys() -> None:
    """Test that numeric strings as keys are quoted."""
    obj = {"1": "one", "2": "two", "normal": "three"}
    result = encode(obj)

    assert '"1": one' in result
    assert '"2": two' in result
    assert "normal: three" in result


def test_encode_boolean_array() -> None:
    """Test encoding array of booleans."""
    result = encode([True, False, True, False])
    assert result == "[4]: true,false,true,false"


def test_encode_null_array() -> None:
    """Test encoding array with all null values."""
    result = encode([None, None, None])
    assert result == "[3]: null,null,null"


def test_encode_object_with_all_types() -> None:
    """Test encoding object containing all supported types."""
    obj = {
        "null": None,
        "bool": True,
        "int": 42,
        "float": 3.14,
        "string": "text",
        "array": [1, 2, 3],
        "object": {"nested": "value"},
    }
    result = encode(obj)

    assert "null: null" in result
    assert "bool: true" in result
    assert "int: 42" in result
    assert "float: 3.14" in result
    assert "string: text" in result
    assert "array[3]: 1,2,3" in result
    assert "object:" in result
    assert "nested: value" in result


def test_encode_dataclass() -> None:
    """Test encoding dataclass instances."""
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int

    person = Person(name="Alice", age=30)
    result = encode(person)

    assert "name: Alice" in result
    assert "age: 30" in result


def test_encode_nested_dataclass() -> None:
    """Test encoding nested dataclass structures."""
    from dataclasses import dataclass

    @dataclass
    class Address:
        city: str
        country: str

    @dataclass
    class Person:
        name: str
        address: Address

    person = Person(name="Bob", address=Address(city="NYC", country="USA"))
    result = encode(person)

    assert "name: Bob" in result
    assert "address:" in result
    assert "city: NYC" in result
    assert "country: USA" in result
