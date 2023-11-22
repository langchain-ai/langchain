"""Toolkit for interacting with a Cube Semantic Layer."""
from typing import List, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.tools.cube.tool import (
    ListCubeTool,
    LoadCubeTool,
    MetaInformationCubeTool,
)
from langchain.utilities.cube import CubeAPIWrapper


class CubeToolkit(BaseToolkit):
    """Toolkit for interacting with Cube Semantic Layer.

    *Security Note*: This toolkit interacts with an external service.

        Control access to who can use this toolkit.

        Make sure that the capabilities given by this toolkit to the calling
        code are appropriately scoped to the application.

        See https://python.langchain.com/docs/security or https://cube.dev/security
         for more information.
    """

    cube: CubeAPIWrapper = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    examples: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        load_cube_tool = LoadCubeTool(cube=self.cube)

        if self.examples is not None and self.examples != "":
            load_cube_tool.description = (
                load_cube_tool.description + f"Examples:\n{self.examples}"
            )

        return [
            load_cube_tool,
            MetaInformationCubeTool(cube=self.cube),
            ListCubeTool(cube=self.cube),
        ]
