from abc import ABC, abstractmethod
from typing import List

from langchain_core.schema.tool import BaseToolInterface


class ToolkitInterface(ABC):
    """Toolkit Interface is a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[BaseToolInterface]:
        """Get the tools in the toolkit."""
