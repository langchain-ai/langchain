"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Tuple, Type, Union

from pydantic import BaseModel

from langchain.tools.structured import BaseStructuredTool


class StringSchema(BaseModel):
    """Schema for a tool with string input."""

    # Child tools can add additional validation by
    # subclassing this schema.
    tool_input: str


class BaseTool(ABC, BaseStructuredTool[str]):
    """Interface LangChain tools must implement."""

    args_schema: Type[StringSchema] = StringSchema  # :meta private:

    def _wrap_input(self, tool_input: Union[str, Dict]) -> Dict:
        """Wrap the tool's input into a pydantic model."""
        if isinstance(tool_input, dict):
            return tool_input
        return {"tool_input": tool_input}

    def _prepare_input(self, input_: dict) -> Tuple[Sequence, Dict]:
        """Prepare the args and kwargs for the tool."""
        # We expect a single string input
        return tuple(input_.values()), {}

    @abstractmethod
    def _run(self, tool_input: str) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: Union[str, dict],
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
        tool_input: Union[str, dict],
        verbose: bool | None = None,
        start_color: str | None = "green",
        color: str | None = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        wrapped_input = self._wrap_input(tool_input)
        return await super().arun(wrapped_input, verbose, start_color, color, **kwargs)

    def __call__(self, tool_input: Union[dict, str]) -> str:
        return self.run(tool_input)
