"""Base implementation for tools or skills."""

from abc import abstractmethod
from pydantic import BaseModel


class Tool(BaseModel):
    """Class responsible for defining a tool or skill for an LLM."""

    name: str
    description: str

    @abstractmethod
    def function(self, *args, **kwargs) -> str:
        """Execute the tool."""
