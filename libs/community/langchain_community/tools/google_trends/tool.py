"""Tool for the Google Trends"""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper


class GoogleTrendsQueryRunToolInput(BaseModel):
    query: str = Field(description="Google Trends query to search with google trends")


class GoogleTrendsQueryRun(BaseTool):
    """Tool that queries the Google trends API."""

    name: str = "google_trends"
    description: str = (
        "A wrapper around Google Trends Search. "
        "Useful for when you need to get information about"
        "google search trends from Google Trends"
        "Input should be a search query."
    )
    api_wrapper: GoogleTrendsAPIWrapper
    args_schema: Type[GoogleTrendsQueryRunToolInput]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
