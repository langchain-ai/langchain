import pytest

from langchain_core.utils.json_schema import dereference_refs


def test_dereference_refs_self_reference_no_recursion() -> None:
    """Ensure self-referential schemas are handled without infinite recursion."""
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "child": {"$ref": "#/$defs/Node"},
                },
            }
        },
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }

    # Should not raise RecursionError and should return a dictionary
    actual = dereference_refs(schema)
    assert isinstance(actual, dict)
    # The $defs should be preserved and recursion should be broken within dereferenced parts
    assert "$defs" in actual
    assert "properties" in actual


def test_dereference_refs_circular_chain_no_recursion() -> None:
    """Ensure multi-node circular chains are handled without infinite recursion."""
    schema = {
        "$defs": {
            "A": {"type": "object", "properties": {"to_b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"to_c": {"$ref": "#/$defs/C"}}},
            "C": {"type": "object", "properties": {"to_a": {"$ref": "#/$defs/A"}}},
        },
        "type": "object",
        "properties": {"start": {"$ref": "#/$defs/A"}},
    }

    # Should not raise RecursionError
    actual = dereference_refs(schema)
    assert isinstance(actual, dict)
    # Spot-check top-level dereference occurred
    assert "properties" in actual
    assert "start" in actual["properties"]
