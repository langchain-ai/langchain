"""Different methods for rendering Tools to be passed to LLMs.

Depending on the LLM you are using and the prompting strategy you are using,
you may want Tools to be rendered in a different way.
This module contains various ways to render tools.
"""
from typing import List

from langchain.tools.base import BaseTool
from langchain.utils.openai_functions import (
    FunctionDescription,
    ToolDescription,
    convert_pydantic_to_openai_function,
)


def render_text_description(tools: List[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


def render_text_description_and_args(tools: List[BaseTool]) -> str:
    """Render the tool name, description, and args in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, \
args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
    return "\n".join(tool_strings)


def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if tool.args_schema:
        return convert_pydantic_to_openai_function(
            tool.args_schema, name=tool.name, description=tool.description
        )
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                # This is a hack to get around the fact that some tools
                # do not expose an args_schema, and expect an argument
                # which is a string.
                # And Open AI does not support an array type for the
                # parameters.
                "properties": {
                    "__arg1": {"title": "__arg1", "type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            },
        }


def format_tool_to_openai_tool(tool: BaseTool) -> ToolDescription:
    """Format tool into the OpenAI function API."""
    function = format_tool_to_openai_function(tool)
    return {"type": "function", "function": function}
