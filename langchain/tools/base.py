"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC
from typing import Type

from pydantic import (
    BaseModel,
)

from langchain.tools.structured import BaseStructuredTool


class BaseTool(ABC, BaseStructuredTool[str, str, str]):
    """Interface LangChain tools must implement."""

    args_schema: Type[str] = str  # :meta private:

    def _parse_input(self, tool_input: str) -> str:
        """Load the tool's input into a pydantic model."""
        return tool_input
