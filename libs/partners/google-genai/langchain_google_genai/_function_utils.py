from __future__ import annotations

from typing import (
    Dict,
    List,
    Type,
    Union,
)

import google.ai.generativelanguage as glm
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.json_schema import dereference_refs

FunctionCallType = Union[BaseTool, Type[BaseModel], Dict]

TYPE_ENUM = {
    "string": glm.Type.STRING,
    "number": glm.Type.NUMBER,
    "integer": glm.Type.INTEGER,
    "boolean": glm.Type.BOOLEAN,
    "array": glm.Type.ARRAY,
    "object": glm.Type.OBJECT,
}


def convert_to_genai_function_declarations(
    function_calls: List[FunctionCallType],
) -> List[glm.Tool]:
    return [
        glm.Tool(
            function_declarations=[_convert_to_genai_function(fc)],
        )
        for fc in function_calls
    ]


def _convert_to_genai_function(fc: FunctionCallType) -> glm.FunctionDeclaration:
    if isinstance(fc, BaseTool):
        return _convert_tool_to_genai_function(fc)
    elif isinstance(fc, type) and issubclass(fc, BaseModel):
        return _convert_pydantic_to_genai_function(fc)
    elif isinstance(fc, dict):
        return glm.FunctionDeclaration(
            name=fc["name"],
            description=fc.get("description"),
            parameters={
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
        )
    else:
        raise ValueError(f"Unsupported function call type {fc}")


def _convert_tool_to_genai_function(tool: BaseTool) -> glm.FunctionDeclaration:
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)

        return glm.FunctionDeclaration(
            name=tool.name or schema["title"],
            description=tool.description or schema["description"],
            parameters={
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
        )
    else:
        return glm.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters={
                "properties": {
                    "__arg1": {"type_": TYPE_ENUM["string"]},
                },
                "required": ["__arg1"],
                "type_": TYPE_ENUM["object"],
            },
        )


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
) -> glm.FunctionDeclaration:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)
    return glm.FunctionDeclaration(
        name=schema["title"],
        description=schema.get("description", ""),
        parameters={
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
    )
