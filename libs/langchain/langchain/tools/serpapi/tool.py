"""Tool for the SerpApi.com API."""

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.utilities.serpapi import SerpAPIWrapper


class SerpApiRun(BaseTool):
    """Tool that queries the SerpApi.com API."""

    name: str = "serpapi"
    description: str = (
        "Fast, easy, and complete API for Google, Bing, and other search engines."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )
    api_wrapper: SerpAPIWrapper = Field(default_factory=SerpAPIWrapper)

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


class SerpApiResults(BaseTool):
    """Tool that queries the SerpApi.com API
    and get back json."""

    name: str = "serpapi_results"
    description: str = (
        "Fast, easy, and complete API for Google, Bing, and other search engines."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )
    api_wrapper: SerpAPIWrapper = Field(default_factory=SerpAPIWrapper)

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
