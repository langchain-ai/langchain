import json
import os
import subprocess
from typing import Optional, Type

from pydantic import BaseModel
from tempfile import TemporaryDirectory
from langchain.output_parsers.base import BaseOutputParser
import logging

logger = logging.getLogger(__name__)

format_instructions = """The output should be a markdown code snippet formatted in the following schema.

```json
{format}
```
Do not include comments in the resulting code snippet.



"""
line_template = """"{name}": {type}"""

MAX_DEPTH = 6


def _get_description_from_prop(prop: dict, depth: int) -> str:
    """Return the optional description of a property."""
    indent = "\t" * depth
    return f"{indent}// {prop['description']}\n" if "description" in prop else ""


def _get_code_line(name: str, type: str, depth: int) -> str:
    """Return the code line for a property."""
    indent = "\t" * depth
    return f"{indent}{line_template.format(name=name, type=type)}"


def _get_type_code_line(name: str, type: str, depth: int) -> str:
    """Return the code line for a property."""
    return _get_code_line(name=name, type=type, depth=depth)  # + ";"


def _get_type_str(v: dict) -> str:
    """Return the type of a property."""
    # TODO: Support all in https://docs.pydantic.dev/usage/schema/#json-schema-types
    if "enum" in v:
        enum_l = v["enum"]
        if isinstance(v["enum"][0], str):
            enum_l = [f'"{e}"' for e in enum_l]
        else:
            enum_l = map(str, enum_l)
        type_str = " | ".join(enum_l) + " "
    elif "type" in v:
        type_str = v["type"]
        if "format" in v:
            type_str = f"{type_str} | {v['format']}"
    else:
        raise RuntimeError(f"Unknown type: {v}")
    return type_str


def _get_sub_string(k: str, v: dict, depth: int) -> str:
    """Return the sub-string for a property."""
    description = _get_description_from_prop(v, depth)
    type_str = _get_type_str(v)
    code = _get_type_code_line(k, type_str, depth)
    return description + code


def resolve_reference(ref: str, definitions: dict) -> dict:
    """Resolve a reference to a definition."""
    if not ref.startswith("#/definitions/"):
        # Arbitrary JSON-schema references not yet supported.
        raise ValueError(f"Unsupported $ref: {ref}")
    ref = ref[len("#/definitions/") :]
    if ref in definitions:
        return definitions[ref]
    else:
        valid_refs = sorted(definitions.keys())
        raise ValueError(f"Unknown reference: {ref}\nExpected one of: {valid_refs}")


def process_reference(k: str, v: dict, depth: int, definitions: dict) -> str:
    """Return the code line for a referred property."""
    resolved = resolve_reference(v["$ref"], definitions)
    nested_type = serialize_schema(resolved, depth=depth + 1, definitions=definitions)
    if nested_type:
        description = _get_description_from_prop(v, depth)
        code = _get_type_code_line(k, nested_type, depth)
        return description + code
    else:
        return _get_sub_string(k, resolved, depth) + "\n"


def serialize_allof(k: str, v: dict, depth: int, definitions: dict) -> str:
    """Return the code lines for an allOf property."""
    result = ""
    for sub_schema in v["allOf"]:
        if "$ref" in sub_schema:
            result += process_reference(k, sub_schema, depth, definitions)
        elif "type" in sub_schema:
            result += _get_type_code_line(k, sub_schema["type"], depth)
        else:
            raise ValueError(f"Unknown type: {sub_schema}")
    return result


def serialize_anyof(k: str, v: dict, depth: int, definitions: dict) -> str:
    """Return the code lines for an anyOf property."""
    result = ""
    for i, sub_schema in enumerate(v["anyOf"]):
        if "$ref" in sub_schema:
            next_opt = process_reference(k, sub_schema, depth, definitions)
        elif "type" in sub_schema:
            next_opt = _get_type_code_line(k, sub_schema["type"], depth)
        else:
            raise ValueError(f"Unknown type: {sub_schema}")
        if i == 0:
            result += next_opt
        else:
            next_opt = next_opt.replace(f'"{k}": ', "", 1)
            result += " | " + next_opt
    return result


def serialize_schema(
    schema: dict, depth: int = 1, definitions: Optional[dict] = None
) -> str:
    """Return the string representation of JSON schema."""
    if depth > MAX_DEPTH:
        logger.error(f"Max depth exceeded: {depth}")
        return ""
    properties = schema.get("properties", {})
    if not properties:
        return ""
    definitions = definitions or schema.get("definitions", {})
    result = "{\n"
    for k, v in properties.items():
        if "$ref" in v:
            result += process_reference(k, v, depth, definitions)
        elif "allOf" in v:
            result += serialize_allof(k, v, depth, definitions)
        elif "anyOf" in v:
            result += serialize_anyof(k, v, depth, definitions)
        else:
            arg_str = _get_sub_string(k, v, depth)
            result += f"{arg_str}\n"
    final_indent = "\t" * (depth - 1)
    result += f"\n{final_indent}}}"
    return result


def parse_json_schema_to_typescript(schema: dict) -> str:
    """Return the string representation of schema."""
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "schema.json"), "w") as f:
            json.dump(schema, f)
        schema_str = subprocess.check_output(
            ["json2ts", "-i", "schema.json"], cwd=tmpdir
        ).decode("utf-8")
    return schema_str


class PydanticOutputParser(BaseOutputParser):
    response_schema: Type[BaseModel]

    def get_format_instructions(self) -> str:
        schema = self.response_schema.schema()
        serialized = serialize_schema(schema)
        return format_instructions.format(format=serialized)

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```json")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        return self.response_schema(**json_obj)
