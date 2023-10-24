"""Toolkit for interacting with a Cube Semantic Layer."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.tools.cube.tool import (
    ListCubeTool,
    InfoCubeTool,
    LoadCubeTool,
)
from langchain.utilities.cube import Cube


class CubeToolkit(BaseToolkit):
    """Toolkit for interacting with Cube Semantic Layer."""

    cube: Cube = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            LoadCubeTool(cube=self.cube),
            InfoCubeTool(cube=self.cube),
            ListCubeTool(cube=self.cube),
        ]
