"""Tool for the Serper.dev Google Search API."""

from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper


class GoogleSerperRun(BaseTool):  # type: ignore[override]
    """Tool that queries the Serper.dev Google search API."""

    name: str = "google_serper"
    description: str = (
        "A low-cost Google Search API."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )
    api_wrapper: GoogleSerperAPIWrapper

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


class GoogleSerperResults(BaseTool):  # type: ignore[override]
    """Tool that queries the Serper.dev Google Search API
    and get back json."""

    name: str = "google_serper_results_json"
    description: str = (
        "A low-cost Google Search API."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON object of the query results"
    )
    api_wrapper: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

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
