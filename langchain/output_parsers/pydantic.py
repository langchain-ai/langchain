import json
from typing import Type

from pydantic import BaseModel

from langchain.output_parsers.base import BaseOutputParser

format_instructions = """The output should be a markdown code snippet formatted in the following schema:

```typescript
{format}
```"""
line_template = """"{name}": {type}"""


def _get_description_from_prop(prop: dict, depth: int) -> str:
    """Return the optional description of a property."""
    indent = "\t" * depth
    return f"{indent}// {prop['description']}\n" if "description" in prop else ""


def _get_code_line(name: str, type: str, depth: int) -> str:
    """Return the code line for a property."""
    indent = "\t" * depth
    return f"{indent}{line_template.format(name=name, type=type)}"


def _get_sub_string(k: str, v: dict, depth: int) -> str:
    """Return the sub-string for a property."""
    description = _get_description_from_prop(v, depth)
    if "type" not in v:
        print(k, v, depth)
    code = _get_code_line(k, v["type"], depth)
    return description + code


def process_nested_schema(k: str, v: dict, depth: int, definitions: dict) -> str:
    ref = v["$ref"]
    if not ref.startswith("#/definitions/"):
        # Arbitrary JSON-schema references not yet supported.
        raise ValueError(f"Unsupported $ref: {ref}")
    ref = ref[len("#/definitions/") :]
    if ref in definitions:
        nested_type = get_nested_schema_str(definitions[ref], depth + 1)
        result += _get_code_line(k, nested_type, depth)
    else:
        raise ValueError(f"Unknown definition {ref}")


def get_nested_schema_str(schema: dict, depth: int = 1) -> str:
    properties = schema.get("properties", {})
    definitions = schema.get("definitions", {})
    result = "{\n"
    for k, v in properties.items():
        if "$ref" in v:
            result += process_nested_schema(
                k, definitions[v["$ref"]], depth, definitions
            )
        elif "allOf" in v:
            # Nested subschema are emitted as an anyOf
            for sub_schema in v["allOf"]:
                if "$ref" in sub_schema:
                    result += process_nested_schema(
                        k, definitions[sub_schema["$ref"]], depth, definitions
                    )
                elif "type" in sub_schema:
                    result += _get_code_line(k, sub_schema["type"], depth)
                else:
                    raise ValueError(f"Unknown type: {sub_schema}")
        elif "anyOf" in v:
            raise NotImplementedError(f"Union types not yet supported: {v}")
        else:
            arg_str = _get_sub_string(k, v, depth)
            result += f"{arg_str}\n"
    final_indent = "\t" * (depth - 1)
    result += f"{final_indent}}}"
    return result


class PydanticOutputParser(BaseOutputParser):
    response_schema: Type[BaseModel]

    def get_format_instructions(self) -> str:
        schema = self.response_schema.schema()
        schema_str = get_nested_schema_str(schema)
        return format_instructions.format(format=schema_str)

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```typescript")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        return self.response_schema(**json_obj)
