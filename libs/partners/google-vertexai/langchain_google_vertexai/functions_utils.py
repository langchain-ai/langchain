from typing import List

from langchain_core.tools import Tool
from langchain_core.utils.json_schema import dereference_refs
from typing_extensions import TypedDict
from vertexai.preview.generative_models import (  # type: ignore
    FunctionDeclaration,
)
from vertexai.preview.generative_models import (
    Tool as VertexTool,  # type: ignore
)


class FunctionDescription(TypedDict):
    """Representation of a callable function to the OpenAI API."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


def _format_tool_to_vertex_function(tool: Tool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)
        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": schema,
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
        function_declarations.append(
            FunctionDeclaration(**_format_tool_to_vertex_function(tool))
        )

    return [VertexTool(function_declarations=function_declarations)]
