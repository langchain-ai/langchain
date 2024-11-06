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
    with pytest.raises(ValueError):
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


def test_dereference_refs_enum_with_description() -> None:
    schema = {
        "$defs": {"Enum": {"enum": ["name", "age"], "title": "Enum", "type": "string"}},
        "properties": {"user": {"$ref": "#/$defs/Enum", "description": "description"}},
        "required": ["user"],
        "type": "object",
    }
    expected = {
        "$defs": {"Enum": {"enum": ["name", "age"], "title": "Enum", "type": "string"}},
        "properties": {
            "user": {
                "description": "description",
                "enum": ["name", "age"],
                "title": "Enum",
                "type": "string",
            }
        },
        "required": ["user"],
        "type": "object",
    }
    actual = dereference_refs(schema)
    assert actual == expected
