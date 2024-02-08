from __future__ import annotations

from typing import (
    Dict,
    List,
    Type,
    Union,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.json_schema import dereference_refs

FunctionCallType = Union[BaseTool, Type[BaseModel], Dict]

TYPE_ENUM = {
    "string": 1,
    "number": 2,
    "integer": 3,
    "boolean": 4,
    "array": 5,
    "object": 6,
}


def convert_to_genai_function_declarations(
    function_calls: List[FunctionCallType],
) -> Dict:
    function_declarations = []
    for fc in function_calls:
        function_declarations.append(_convert_to_genai_function(fc))
    return {
        "function_declarations": function_declarations,
    }


def _convert_to_genai_function(fc: FunctionCallType) -> Dict:
    """
        Produce

        {
      "name": "get_weather",
      "description": "Determine weather in my location",
      "parameters": {
        "properties": {
          "location": {
            "description": "The city and state e.g. San Francisco, CA",
            "type_": 1
          },
          "unit": { "enum": ["c", "f"], "type_": 1 }
        },
        "required": ["location"],
        "type_": 6
      }
    }

    """
    if isinstance(fc, BaseTool):
        return _convert_tool_to_genai_function(fc)
    elif isinstance(fc, type) and issubclass(fc, BaseModel):
        return _convert_pydantic_to_genai_function(fc)
    elif isinstance(fc, dict):
        return {
            **fc,
            "parameters": {
                "properties": {
                    k: {
                        "type_": TYPE_ENUM[v["type"]],
                        "description": v.get("description"),
                    }
                    for k, v in fc["parameters"]["properties"].items()
                },
                "required": fc["parameters"].get("required", []),
                "type_": TYPE_ENUM[fc["parameters"]["type"]],
            },
        }
    else:
        raise ValueError(f"Unsupported function call type {fc}")


def _convert_tool_to_genai_function(tool: BaseTool) -> Dict:
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)

        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": {
                "properties": {
                    k: {
                        "type_": TYPE_ENUM[v["type"]],
                        "description": v.get("description"),
                    }
                    for k, v in schema["properties"].items()
                },
                "required": schema["required"],
                "type_": TYPE_ENUM[schema["type"]],
            },
        }
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "properties": {
                    "__arg1": {"type": "string"},
                },
                "required": ["__arg1"],
                "type_": TYPE_ENUM["object"],
            },
        }


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
) -> Dict:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)

    return {
        "name": schema["title"],
        "description": schema.get("description", ""),
        "parameters": {
            "properties": {
                k: {
                    "type_": TYPE_ENUM[v["type"]],
                    "description": v.get("description"),
                }
                for k, v in schema["properties"].items()
            },
            "required": schema["required"],
            "type_": TYPE_ENUM[schema["type"]],
        },
    }
