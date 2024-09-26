"""Tool for the SearchApi.io search API."""

from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.searchapi import SearchApiAPIWrapper


class SearchAPIRun(BaseTool):
    """Tool that queries the SearchApi.io search API."""

    name: str = "searchapi"
    description: str = (
        "Google search API provided by SearchApi.io."
        "This tool is handy when you need to answer questions about current events."
        "Input should be a search query."
    )
    api_wrapper: SearchApiAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return await self.api_wrapper.arun(query)


class SearchAPIResults(BaseTool):
    """Tool that queries the SearchApi.io search API and returns JSON."""

    name: str = "searchapi_results_json"
    description: str = (
        "Google search API provided by SearchApi.io."
        "This tool is handy when you need to answer questions about current events."
        "The input should be a search query and the output is a JSON object "
        "with the query results."
    )
    api_wrapper: SearchApiAPIWrapper = Field(default_factory=SearchApiAPIWrapper)

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
