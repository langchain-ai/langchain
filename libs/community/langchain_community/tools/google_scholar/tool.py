"""Tool for the Google Scholar"""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper


class GoogleScholarQueryRunInput(BaseModel):
    """Input for the GoogleScholarQueryRun tool."""

    query: str = Field(description="Query for Google Scholar")


class GoogleScholarQueryRun(BaseTool):
    """Tool that queries the Google search API."""

    name: str = "google_scholar"
    description: str = (
        "A wrapper around Google Scholar Search. "
        "Useful for when you need to get information about"
        "research papers from Google Scholar"
        "Input should be a search query."
    )
    api_wrapper: GoogleScholarAPIWrapper
    args_schema: Type[BaseModel] = GoogleScholarQueryRunInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
