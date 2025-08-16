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


def test_dereference_refs_mixed_ref_with_properties() -> None:
    """Test dereferencing refs that have $ref plus other properties."""
    # This pattern can cause infinite recursion if not handled correctly
    schema = {
        "type": "object",
        "properties": {
            "data": {
                "$ref": "#/$defs/BaseType",
                "description": "Additional description",
                "example": "some example",
            }
        },
        "$defs": {"BaseType": {"type": "string", "minLength": 1}},
    }

    expected = {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "minLength": 1,
                "description": "Additional description",
                "example": "some example",
            }
        },
        "$defs": {"BaseType": {"type": "string", "minLength": 1}},
    }

    actual = dereference_refs(schema)
    assert actual == expected


def test_dereference_refs_complex_apollo_mcp_pattern() -> None:
    """Test pattern that caused infinite recursion in Apollo MCP server schemas."""
    # Simplified version of the problematic pattern from the issue
    schema = {
        "type": "object",
        "properties": {
            "query": {"$ref": "#/$defs/Query", "additionalProperties": False}
        },
        "$defs": {
            "Query": {
                "type": "object",
                "properties": {"user": {"$ref": "#/$defs/User"}},
            },
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "profile": {"$ref": "#/$defs/UserProfile", "nullable": True},
                },
            },
            "UserProfile": {
                "type": "object",
                "properties": {"bio": {"type": "string"}},
            },
        },
    }

    # This should not cause infinite recursion
    actual = dereference_refs(schema)

    # The mixed $ref should be properly resolved and merged
    expected_user_profile = {
        "type": "object",
        "properties": {"bio": {"type": "string"}},
    }

    expected_user = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "profile": {
                "type": "object",
                "properties": {"bio": {"type": "string"}},
                "nullable": True,
            },
        },
    }

    expected_query = {
        "type": "object",
        "properties": {"user": expected_user},
        "additionalProperties": False,
    }

    expected = {
        "type": "object",
        "properties": {"query": expected_query},
        "$defs": {
            "Query": {
                "type": "object",
                "properties": {"user": {"$ref": "#/$defs/User"}},
            },
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "profile": {"$ref": "#/$defs/UserProfile", "nullable": True},
                },
            },
            "UserProfile": expected_user_profile,
        },
    }

    assert actual == expected


def test_dereference_refs_cyclical_mixed_refs() -> None:
    """Test cyclical references with mixed $ref properties don't cause loops."""
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "parent": {"$ref": "#/$defs/Node", "nullable": True},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }
        },
    }

    # This should handle cycles gracefully
    actual = dereference_refs(schema)

    # The self-referencing should be broken with empty objects

    # Verify the structure is correct and doesn't cause infinite recursion
    assert "properties" in actual
    assert "node" in actual["properties"]
    assert isinstance(actual["properties"]["node"], dict)
    assert "type" in actual["properties"]["node"]
