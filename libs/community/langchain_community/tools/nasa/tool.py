"""
This tool allows agents to interact with the NASA API, specifically 
the the NASA Image & Video Library and Exoplanet
"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.nasa import NasaAPIWrapper


class NasaAction(BaseTool):
    """Tool that queries the Atlassian Jira API."""

    api_wrapper: NasaAPIWrapper = Field(default_factory=NasaAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the NASA API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)
