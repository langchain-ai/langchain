"""Tool for the Google Trends"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper


class GoogleJobsQueryRun(BaseTool):
    """Tool that queries the Google Jobs API."""

    name: str = "google_jobs"
    description: str = (
        "A wrapper around Google Jobs Search. "
        "Useful for when you need to get information about"
        "google search Jobs from Google Jobs"
        "Input should be a search query."
    )
    api_wrapper: GoogleJobsAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
