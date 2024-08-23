"""Helper utilities for pydantic.

This module includes helper utilities to ease the migration from pydantic v1 to v2.

They're meant to be used in the following way:

1) Use utility code to help (selected) unit tests pass without modifications
2) Upgrade the unit tests to match pydantic 2
3) Stop using the utility code
"""

from typing import Any


# Function to replace allOf with $ref
def _replace_all_of_with_ref(schema: Any) -> None:
    """Replace allOf with $ref in the schema."""
    if isinstance(schema, dict):
        # If the schema has an allOf key with a single item that contains a $ref
        if (
            "allOf" in schema
            and len(schema["allOf"]) == 1
            and "$ref" in schema["allOf"][0]
        ):
            schema["$ref"] = schema["allOf"][0]["$ref"]
            del schema["allOf"]
            if "default" in schema and schema["default"] is None:
                del schema["default"]
        else:
            # Recursively process nested schemas
            for key, value in schema.items():
                if isinstance(value, (dict, list)):
                    _replace_all_of_with_ref(value)
    elif isinstance(schema, list):
        for item in schema:
            _replace_all_of_with_ref(item)


def _remove_bad_none_defaults(schema: Any) -> None:
    """Removing all none defaults.

    Pydantic v1 did not generate these, but Pydantic v2 does.

    The None defaults usually represent **NotRequired** fields, and the None value
    is actually **incorrect** as a value since the fields do not allow a None value.

    See difference between Optional and NotRequired types in python.
    """
    if isinstance(schema, dict):
        for key, value in schema.items():
            if isinstance(value, dict):
                if "default" in value and value["default"] is None:
                    any_of = value.get("anyOf", [])
                    for type_ in any_of:
                        if "type" in type_ and type_["type"] == "null":
                            break  # Null type explicitly defined
                    else:
                        del value["default"]
                _remove_bad_none_defaults(value)
            elif isinstance(value, list):
                for item in value:
                    _remove_bad_none_defaults(item)
    elif isinstance(schema, list):
        for item in schema:
            _remove_bad_none_defaults(item)


def _schema(obj: Any) -> dict:
    """Get the schema of a pydantic model in the pydantic v1 style.

    This will attempt to map the schema as close as possible to the pydantic v1 schema.
    """
    # Remap to old style schema
    if not hasattr(obj, "model_json_schema"):  # V1 model
        return obj.schema()

    # Then we're using V2 models internally.
    raise AssertionError(
        "Hi there! Looks like you're attempting to upgrade to Pydantic v2. If so: \n"
        "1) remove this exception\n"
        "2) confirm that the old unit tests pass, and if not look for difference\n"
        "3) update the unit tests to match the new schema\n"
        "4) remove this utility function\n"
    )

    schema_ = obj.model_json_schema(ref_template="#/definitions/{model}")
    if "$defs" in schema_:
        schema_["definitions"] = schema_["$defs"]
        del schema_["$defs"]

    _replace_all_of_with_ref(schema_)
    _remove_bad_none_defaults(schema_)

    return schema_
