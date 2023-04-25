"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Type, Union


from langchain.tools.structured import BaseStructuredTool


class BaseTool(ABC, BaseStructuredTool[str, str]):
    """Interface LangChain tools must implement."""

    args_schema: Type[str] = str  # :meta private:

    def _parse_input(self, tool_input: Dict) -> str:
        """Load the tool's input into a pydantic model."""
        if len(tool_input) == 1:
            # Make base tools more forwards compatible
            result = next(iter(tool_input.values()))
            if not isinstance(result, str):
                raise ValueError(
                    f"Tool input {tool_input} must be a single string or dict."
                )
            return result
        raise ValueError(f"Tool input {tool_input} must be a single string or dict.")

    def _wrap_input(self, tool_input: Union[str, Dict]) -> Dict:
        """Wrap the tool's input into a pydantic model."""
        if isinstance(tool_input, str):
            return {"tool_input": tool_input}
        else:
            return tool_input

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: bool | None = None,
        start_color: str | None = "green",
        color: str | None = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        wrapped_input = self._wrap_input(tool_input)
        return super().run(wrapped_input, verbose, start_color, color, **kwargs)

    async def arun(
        self,
        tool_input: Union[str, Dict],
        verbose: bool | None = None,
        start_color: str | None = "green",
        color: str | None = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        wrapped_input = self._wrap_input(tool_input)
        return await super().arun(wrapped_input, verbose, start_color, color, **kwargs)
