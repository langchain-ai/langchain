"""Utils for parsing OpenAPI specs to typescript strings."""
import logging
import re
from typing import Union

from langchain.chains.api.openapi.typed_parser import resolve_schema
from openapi_schema_pydantic import OpenAPI, Reference, Schema

logger = logging.getLogger(__name__)


def add_description_and_example(prop_schema: Schema) -> str:
    """Add a description and example to a typescript property."""
    description = prop_schema.description or ""
    example = prop_schema.example
    if example is not None:
        description += f" Example: {repr(example)}"
    description = description.strip()
    if description:
        return f"/* {description} */\n"
    return ""


def object_type_to_typescript(schema: Schema, spec: OpenAPI, verbose: bool) -> str:
    """Convert schema of type 'object' to a typescript string."""
    properties = []
    required_props = set(schema.required or [])
    if schema.properties:
        for prop_name, prop_schema in schema.properties.items():
            if not prop_name.strip():
                continue
            illegal_ts_identifier_chars = r"[.\- :;,?/]"
            if re.search(illegal_ts_identifier_chars, prop_name):
                prop_name = f"'{prop_name}'"
            if isinstance(prop_schema, Reference):
                prop_schema = resolve_schema(prop_schema, spec)
            prop_type = schema_to_typescript(prop_schema, spec, verbose)
            optional = "" if prop_name in required_props else "?"
            description = add_description_and_example(prop_schema) if verbose else ""
            properties.append(f"{description}{prop_name}{optional}: {prop_type}")
        return "{\n" + ",\n".join(properties) + "\n}"
    elif schema.additionalProperties:
        if schema.additionalProperties is not True:
            # if additionaProperties is Reference | Schema
            additional_type = schema_to_typescript(schema.additionalProperties, spec)
            return f"{{ [key: string]: {additional_type} }}"
        else:
            # If additionalProperties is Literal[True] then it means
            # additional properties can be specified but no further
            # information is provided.
            return "any"
    else:
        return "{}"


def schema_to_typescript(
    schema: Union[Schema, Reference], spec: OpenAPI, verbose: bool = False
) -> str:
    """Convert pydantic Schema object to a typescript string."""
    if isinstance(schema, Reference):
        schema = resolve_schema(schema, spec)
    if schema.type == "object":
        return object_type_to_typescript(schema, spec, verbose)
    elif schema.type == "array":
        item_type = schema_to_typescript(schema.items, spec)
        return f"Array<{item_type}>"
    elif schema.type in ("integer", "number"):
        return "number"
    elif schema.type in ("string", "boolean", "null"):
        return schema.type
    else:
        return "unknown"
