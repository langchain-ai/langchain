"""Different methods for rendering Tools to be passed to LLMs.

Depending on the LLM you are using and the prompting strategy you are using,
you may want Tools to be rendered in a different way.
This module contains various ways to render tools.
"""
from typing import Callable, List

# For backwards compatibility
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    format_tool_to_openai_function,
    format_tool_to_openai_tool,
)

__all__ = [
    "ToolsRenderer",
    "render_text_description",
    "render_text_description_and_args",
    "format_tool_to_openai_tool",
    "format_tool_to_openai_function",
]


ToolsRenderer = Callable[[List[BaseTool]], str]


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
