from typing import TYPE_CHECKING

from langchain.agents import Tool

from langchain_community.utils.openai_functions import (
    FunctionDescription,
    convert_pydantic_to_openai_function,
)

if TYPE_CHECKING:
    from vertexai.preview.generative_models import Tool as VertexTool


def format_tool_to_vertex_function(tool: Tool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        return convert_pydantic_to_openai_function(
            tool.args_schema, name=tool.name, description=tool.description
        )
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


def format_tool_to_vertex_tool(tool: Tool) -> "VertexTool":
    "Format tool into the Vertex Tool instance."

    from vertexai.preview.generative_models import FunctionDeclaration
    from vertexai.preview.generative_models import Tool as VertexTool

    function_declaration = FunctionDeclaration(**format_tool_to_vertex_function(tool))
    return VertexTool(function_declarations=[function_declaration])
