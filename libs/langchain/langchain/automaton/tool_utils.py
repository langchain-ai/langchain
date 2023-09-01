from typing import Sequence, TypedDict

from langchain.tools import Tool


def _generate_tools_descriptions(tools: Sequence[Tool]) -> str:
    """Generate a description of the tools."""
    return "\n".join([f"{tool_.name}: {tool_.description}" for tool_ in tools]) + "\n"


class ToolInfo(TypedDict):
    """A dictionary containing information about a tool."""

    tool_names: str
    tools_description: str


def generate_tool_info(tools: Sequence[Tool]) -> ToolInfo:
    """Generate a string containing the names of the tools and their descriptions."""
    tools_description = _generate_tools_descriptions(tools)
    tool_names = ", ".join([tool_.name for tool_ in tools])
    return {
        "tool_names": tool_names,
        "tools_description": tools_description,
    }
