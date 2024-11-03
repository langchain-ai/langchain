"""Tool for the Google Trends"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper


class GoogleTrendsQueryRun(BaseTool):  # type: ignore[override]
    """Tool that queries the Google trends API."""

    name: str = "google_trends"
    description: str = (
        "A wrapper around Google Trends Search. "
        "Useful for when you need to get information about"
        "google search trends from Google Trends"
        "Input should be a search query."
    )
    api_wrapper: GoogleTrendsAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
