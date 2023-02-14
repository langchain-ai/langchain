"""Toolkits form collections of tools that can be executed by an LLM."""


from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from langchain.tools.tool import Tool


class Toolkit(BaseModel):
    """Class responsible for defining a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """Get the tools in the toolkit."""
