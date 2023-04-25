"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC
from typing import Dict, Type, Union


from langchain.tools.structured import AbstractBaseTool


class BaseTool(ABC, AbstractBaseTool[Union[dict, str], str, str]):
    """Interface LangChain tools must implement."""

    args_schema: Type[str] = str  # :meta private:

    def _parse_input(self, tool_input: Union[dict, str]) -> str:
        """Load the tool's input into a pydantic model."""
        if isinstance(tool_input, str):
            return tool_input
        if len(tool_input) == 1:
            result = next(iter(tool_input.values()))
            if not isinstance(result, str):
                raise ValueError(
                    f"Tool input {tool_input} must be a single string or dict."
                )
            return result
        raise ValueError(f"Tool input {tool_input} must be a single string or dict.")
