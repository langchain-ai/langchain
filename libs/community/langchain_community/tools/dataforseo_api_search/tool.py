"""Tool for the DataForSeo SERP API."""

from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper


class DataForSeoAPISearchRun(BaseTool):  # type: ignore[override]
    """Tool that queries the DataForSeo Google search API."""

    name: str = "dataforseo_api_search"
    description: str = (
        "A robust Google Search API provided by DataForSeo."
        "This tool is handy when you need information about trending topics "
        "or current events."
    )
    api_wrapper: DataForSeoAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.api_wrapper.run(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return (await self.api_wrapper.arun(query)).__str__()


class DataForSeoAPISearchResults(BaseTool):  # type: ignore[override]
    """Tool that queries the DataForSeo Google Search API
    and get back json."""

    name: str = "dataforseo_results_json"
    description: str = (
        "A comprehensive Google Search API provided by DataForSeo."
        "This tool is useful for obtaining real-time data on current events "
        "or popular searches."
        "The input should be a search query and the output is a JSON object "
        "of the query results."
    )
    api_wrapper: DataForSeoAPIWrapper = Field(default_factory=DataForSeoAPIWrapper)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.api_wrapper.results(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return (await self.api_wrapper.aresults(query)).__str__()
