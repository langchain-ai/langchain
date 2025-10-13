from typing import Any

from pydantic import BaseModel

from langchain_core.utils.pydantic import is_basemodel_subclass


# Function to replace allOf with $ref
def replace_all_of_with_ref(schema: Any) -> None:
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
            for value in schema.values():
                if isinstance(value, (dict, list)):
                    replace_all_of_with_ref(value)
    elif isinstance(schema, list):
        for item in schema:
            replace_all_of_with_ref(item)


def remove_all_none_default(schema: Any) -> None:
    """Removing all none defaults.

    Pydantic v1 did not generate these, but Pydantic v2 does.

    The None defaults usually represent **NotRequired** fields, and the None value
    is actually **incorrect** as a value since the fields do not allow a None value.

    See difference between Optional and NotRequired types in python.
    """
    if isinstance(schema, dict):
        for value in schema.values():
            if isinstance(value, dict):
                if "default" in value and value["default"] is None:
                    any_of = value.get("anyOf", [])
                    for type_ in any_of:
                        if "type" in type_ and type_["type"] == "null":
                            break  # Null type explicitly defined
                    else:
                        del value["default"]
                remove_all_none_default(value)
            elif isinstance(value, list):
                for item in value:
                    remove_all_none_default(item)
    elif isinstance(schema, list):
        for item in schema:
            remove_all_none_default(item)


def _remove_enum(obj: Any) -> None:
    """Remove the description from enums."""
    if isinstance(obj, dict):
        if "enum" in obj:
            if "description" in obj and obj["description"] == "An enumeration.":
                del obj["description"]
            if "type" in obj and obj["type"] == "string":
                del obj["type"]
            del obj["enum"]
        for value in obj.values():
            _remove_enum(value)
    elif isinstance(obj, list):
        for item in obj:
            _remove_enum(item)


def _schema(obj: Any) -> dict:
    """Return the schema of the object."""
    if not is_basemodel_subclass(obj):
        msg = f"Object must be a Pydantic BaseModel subclass. Got {type(obj)}"
        raise TypeError(msg)
    # Remap to old style schema
    if not hasattr(obj, "model_json_schema"):  # V1 model
        return obj.schema()

    schema_ = obj.model_json_schema(ref_template="#/definitions/{model}")
    if "$defs" in schema_:
        schema_["definitions"] = schema_["$defs"]
        del schema_["$defs"]

    if "default" in schema_ and schema_["default"] is None:
        del schema_["default"]

    replace_all_of_with_ref(schema_)
    remove_all_none_default(schema_)
    _remove_enum(schema_)

    return schema_


def _remove_additionalproperties_from_untyped_dicts(schema: dict) -> dict[str, Any]:
    """Remove `"additionalProperties": True` from dicts in the schema.

    Pydantic 2.11 and later versions include `"additionalProperties": True` when
    generating JSON schemas for dict properties with `Any` or `object` values.
    """

    def _remove_dict_additional_props(
        obj: dict[str, Any] | list[Any], *, inside_properties: bool = False
    ) -> None:
        if isinstance(obj, dict):
            if (
                inside_properties
                and obj.get("type") == "object"
                and obj.get("additionalProperties") is True
            ):
                obj.pop("additionalProperties", None)

            # Recursively scan children
            for key, value in obj.items():
                # We are "inside_properties" if the *current* key is "properties",
                # or if we were already inside properties in the caller.
                next_inside_properties = inside_properties or (key == "properties")
                _remove_dict_additional_props(
                    value, inside_properties=next_inside_properties
                )

        elif isinstance(obj, list):
            for item in obj:
                _remove_dict_additional_props(item, inside_properties=inside_properties)

    _remove_dict_additional_props(schema, inside_properties=False)
    return schema


def _normalize_schema(obj: Any) -> dict[str, Any]:
    """Generate a schema and normalize it.

    This will collapse single element allOfs into $ref.

    For example,

    'obj': {'allOf': [{'$ref': '#/$defs/obj'}]

    to:

    'obj': {'$ref': '#/$defs/obj'}

    Args:
        obj: The object to generate the schema for
    """
    data = obj.model_json_schema() if isinstance(obj, BaseModel) else obj
    remove_all_none_default(data)
    replace_all_of_with_ref(data)
    _remove_enum(data)
    _remove_additionalproperties_from_untyped_dicts(data)
    return data
