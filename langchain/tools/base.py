"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from langchain.tools.structured import AbstractStructuredTool


class BaseTool(ABC, AbstractStructuredTool[str, str]):
    """Interface LangChain tools must implement."""

    args_schema: Type[str] = str  # :meta private:

    def _wrap_input(self, tool_input: Union[str, Dict]) -> Dict:
        """Wrap the tool's input into a pydantic model."""
        if isinstance(tool_input, Dict):
            return tool_input
        return {"tool_input": tool_input}

    def _parse_input(self, input_: Dict) -> str:
        """Prepare the args and kwargs for the tool."""
        return next(iter(input_.values()))

    @property
    def args(self) -> Dict:
        """Return the JSON schema for the tool's args."""
        return {"properties": {"tool_input": {"type": "string"}}}

    @abstractmethod
    def _run(self, tool_input: str) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        wrapped_input = self._wrap_input(tool_input)
        return super().run(wrapped_input, verbose, start_color, color, **kwargs)

    async def arun(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        wrapped_input = self._wrap_input(tool_input)
        return await super().arun(wrapped_input, verbose, start_color, color, **kwargs)

    def __call__(self, tool_input: Union[str, Dict]) -> str:
        return self.run(tool_input)
