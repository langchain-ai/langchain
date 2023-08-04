"""Toolkits for agents."""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from langchain.tools import BaseTool


class BaseToolkit(BaseModel, ABC):
    """Base Toolkit representing a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
