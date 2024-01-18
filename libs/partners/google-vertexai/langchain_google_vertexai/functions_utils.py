from typing import List

from langchain_core.tools import Tool
from langchain_core.utils.function_calling import FunctionDescription
from langchain_core.utils.json_schema import dereference_refs
from vertexai.preview.generative_models import (  # type: ignore
    FunctionDeclaration,
)
from vertexai.preview.generative_models import (
    Tool as VertexTool,  # type: ignore
)


def _format_tool_to_vertex_function(tool: Tool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)

        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": {
                "properties": {
                    k: {
                        "type": v["type"],
                        "description": v.get("description"),
                    }
                    for k, v in schema["properties"].items()
                },
                "required": schema["required"],
                "type": schema["type"],
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
                "type": "object",
            },
        }


def _format_tools_to_vertex_tool(tools: List[Tool]) -> List[VertexTool]:
    "Format tool into the Vertex Tool instance."
    function_declarations = []
    for tool in tools:
        func = _format_tool_to_vertex_function(tool)
        function_declarations.append(FunctionDeclaration(**func))

    return [VertexTool(function_declarations=function_declarations)]
