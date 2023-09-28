from typing import Sequence

from langchain.tools.base import BaseTool


def validate_tools_single_input(class_name: str, tools: Sequence[BaseTool]) -> None:
    """Validate tools for single input."""
    for tool in tools:
        if not tool.is_single_input:
            raise ValueError(
                f"{class_name} does not support multi-input tool {tool.name}."
            )
