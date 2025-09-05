import pytest

from langchain_core.utils.json_schema import dereference_refs


def test_dereference_refs_no_refs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
        },
    }
    actual = dereference_refs(schema)
    assert actual == schema


def test_dereference_refs_one_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "first_name": {"$ref": "#/$defs/name"},
        },
        "$defs": {"name": {"type": "string"}},
    }
    expected = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
        },
        "$defs": {"name": {"type": "string"}},
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_multiple_refs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "first_name": {"$ref": "#/$defs/name"},
            "other": {"$ref": "#/$defs/other"},
        },
        "$defs": {
            "name": {"type": "string"},
            "other": {"type": "object", "properties": {"age": "int", "height": "int"}},
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "other": {"type": "object", "properties": {"age": "int", "height": "int"}},
        },
        "$defs": {
            "name": {"type": "string"},
            "other": {"type": "object", "properties": {"age": "int", "height": "int"}},
        },
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_nested_refs_skip() -> None:
    schema = {
        "type": "object",
        "properties": {
            "info": {"$ref": "#/$defs/info"},
        },
        "$defs": {
            "name": {"type": "string"},
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"$ref": "#/$defs/name"}},
            },
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"type": "string"}},
            },
        },
        "$defs": {
            "name": {"type": "string"},
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"$ref": "#/$defs/name"}},
            },
        },
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_nested_refs_no_skip() -> None:
    schema = {
        "type": "object",
        "properties": {
            "info": {"$ref": "#/$defs/info"},
        },
        "$defs": {
            "name": {"type": "string"},
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"$ref": "#/$defs/name"}},
            },
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"type": "string"}},
            },
        },
        "$defs": {
            "name": {"type": "string"},
            "info": {
                "type": "object",
                "properties": {"age": "int", "name": {"type": "string"}},
            },
        },
    }
    actual = dereference_refs(schema, skip_keys=())
    assert actual == expected


def test_dereference_refs_missing_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "first_name": {"$ref": "#/$defs/name"},
        },
        "$defs": {},
    }
    with pytest.raises(KeyError):
        dereference_refs(schema)


def test_dereference_refs_remote_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "first_name": {"$ref": "https://somewhere/else/name"},
        },
    }
    with pytest.raises(ValueError, match="ref paths are expected to be URI fragments"):
        dereference_refs(schema)


def test_dereference_refs_integer_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "error_400": {"$ref": "#/$defs/400"},
        },
        "$defs": {
            400: {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "error_400": {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
        "$defs": {
            400: {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_string_ref() -> None:
    schema = {
        "type": "object",
        "properties": {
            "error_400": {"$ref": "#/$defs/400"},
        },
        "$defs": {
            "400": {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "error_400": {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
        "$defs": {
            "400": {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_cyclical_refs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "user": {"$ref": "#/$defs/user"},
            "customer": {"$ref": "#/$defs/user"},
        },
        "$defs": {
            "user": {
                "type": "object",
                "properties": {
                    "friends": {"type": "array", "items": {"$ref": "#/$defs/user"}}
                },
            }
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "friends": {
                        "type": "array",
                        "items": {},  # Recursion is broken here
                    }
                },
            },
            "customer": {
                "type": "object",
                "properties": {
                    "friends": {
                        "type": "array",
                        "items": {},  # Recursion is broken here
                    }
                },
            },
        },
        "$defs": {
            "user": {
                "type": "object",
                "properties": {
                    "friends": {"type": "array", "items": {"$ref": "#/$defs/user"}}
                },
            }
        },
    }
    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_list_index() -> None:
    """Test dereferencing refs that use list indices (e.g., anyOf/1)."""
    # Test case from the issue report - anyOf array with numeric index reference
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "anyOf": [
                    {  # variant 0
                        "type": "object",
                        "properties": {"kind": {"type": "string", "const": "ONE"}},
                    },
                    {  # variant 1
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "TWO"},
                            "startDate": {
                                "type": "string",
                                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                            },
                            "endDate": {
                                "$ref": (
                                    "#/properties/payload/anyOf/1/properties/startDate"
                                )
                            },
                        },
                    },
                ]
            }
        },
    }

    expected = {
        "type": "object",
        "properties": {
            "payload": {
                "anyOf": [
                    {  # variant 0
                        "type": "object",
                        "properties": {"kind": {"type": "string", "const": "ONE"}},
                    },
                    {  # variant 1
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "TWO"},
                            "startDate": {
                                "type": "string",
                                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                            },
                            "endDate": {
                                "type": "string",
                                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                            },
                        },
                    },
                ]
            }
        },
    }

    actual = dereference_refs(schema)
    assert actual == expected

    # Test oneOf array with numeric index reference
    schema_oneof = {
        "type": "object",
        "properties": {
            "data": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {
                        "type": "object",
                        "properties": {"value": {"$ref": "#/properties/data/oneOf/1"}},
                    },
                ]
            }
        },
    }

    expected_oneof = {
        "type": "object",
        "properties": {
            "data": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "object", "properties": {"value": {"type": "number"}}},
                ]
            }
        },
    }

    actual_oneof = dereference_refs(schema_oneof)
    assert actual_oneof == expected_oneof

    # Test allOf array with numeric index reference
    schema_allof = {
        "type": "object",
        "allOf": [
            {"properties": {"name": {"type": "string"}}},
            {"properties": {"age": {"type": "number"}}},
        ],
        "properties": {"copy_name": {"$ref": "#/allOf/0/properties/name"}},
    }

    expected_allof = {
        "type": "object",
        "allOf": [
            {"properties": {"name": {"type": "string"}}},
            {"properties": {"age": {"type": "number"}}},
        ],
        "properties": {"copy_name": {"type": "string"}},
    }

    actual_allof = dereference_refs(schema_allof)
    assert actual_allof == expected_allof

    # Test edge case: out-of-bounds index should raise KeyError
    schema_invalid = {
        "type": "object",
        "properties": {
            "data": {"anyOf": [{"type": "string"}]},
            "invalid": {"$ref": "#/properties/data/anyOf/5"},  # Index 5 doesn't exist
        },
    }

    with pytest.raises(
        KeyError, match="Reference '#/properties/data/anyOf/5' not found"
    ):
        dereference_refs(schema_invalid)

    # Test edge case: negative index should raise KeyError
    schema_negative = {
        "type": "object",
        "properties": {
            "data": {"anyOf": [{"type": "string"}]},
            "invalid": {"$ref": "#/properties/data/anyOf/-1"},  # Negative index
        },
    }

    with pytest.raises(
        KeyError, match="Reference '#/properties/data/anyOf/-1' not found"
    ):
        dereference_refs(schema_negative)

    # Test that existing dictionary-based numeric key functionality still works
    schema_dict_key = {
        "type": "object",
        "properties": {
            "error_400": {"$ref": "#/$defs/400"},
        },
        "$defs": {
            400: {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }

    expected_dict_key = {
        "type": "object",
        "properties": {
            "error_400": {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
        "$defs": {
            400: {
                "type": "object",
                "properties": {"description": "Bad Request"},
            },
        },
    }

    actual_dict_key = dereference_refs(schema_dict_key)
    assert actual_dict_key == expected_dict_key


def test_dereference_refs_three_node_cycle_mixed_refs() -> None:
    """Ensure long cycles with mixed $ref properties terminate and preserve extras."""
    schema = {
        "type": "object",
        "properties": {
            "root": {"$ref": "#/$defs/A", "description": "root node"},
        },
        "$defs": {
            "A": {
                "type": "object",
                "properties": {"b": {"$ref": "#/$defs/B", "nullable": True}},
            },
            "B": {
                "type": "object",
                "properties": {"c": {"$ref": "#/$defs/C"}},
            },
            "C": {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
            },
        },
    }

    actual = dereference_refs(schema)

    # Basic shape checks (don’t assert deep equality to avoid over-constraining):
    assert isinstance(actual, dict)
    assert actual.get("type") == "object"
    assert "properties" in actual and "root" in actual["properties"]
    root = actual["properties"]["root"]
    assert isinstance(root, dict) and root.get("type") == "object"
    # Verify our extra property on the mixed $ref made it through
    assert root.get("description") == "root node"

    # Follow through one hop and check the 'nullable' flag survived
    a_props = root.get("properties", {})
    b_obj = a_props.get("b")
    assert isinstance(b_obj, dict)
    assert b_obj.get("nullable") is True


def test_dereference_refs_ref_inside_allof_preserves_other_parts() -> None:
    """Ensure $ref inside `allOf` is dereferenced and merged without losing sibling items.
    This guards against regressions with mixed $ref + composition keywords.
    """
    schema = {
        "type": "object",
        "properties": {"container": {"$ref": "#/$defs/Container"}},
        "$defs": {
            "Meta": {"type": "object", "properties": {"id": {"type": "string"}}},
            "Item": {
                "type": "object",
                "properties": {"container": {"$ref": "#/$defs/Container"}},
            },
            "Container": {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Item"},
                            }
                        },
                    },
                    {"$ref": "#/$defs/Meta"},
                ]
            },
        },
    }

    actual = dereference_refs(schema)
    assert "properties" in actual and "container" in actual["properties"]
    container = actual["properties"]["container"]

    # The container schema should still be an allOf with two parts
    assert isinstance(container, dict)
    assert "allOf" in container and isinstance(container["allOf"], list)
    parts = container["allOf"]
    assert len(parts) == 2

    # One part should have 'items' (array of Item), the other should be the dereferenced Meta
    has_items = any(
        isinstance(p, dict) and "properties" in p and "items" in p["properties"]
        for p in parts
    )
    has_meta = any(
        isinstance(p, dict) and "properties" in p and "id" in p["properties"]
        for p in parts
    )
    assert has_items and has_meta
