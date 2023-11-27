"""Tool for the SearxNG search API."""
from typing import Optional

from langchain_core.pydantic_v1 import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.utilities.searx_search import SearxSearchWrapper


class SearxSearchRun(BaseTool):
    """Tool that queries a Searx instance."""

    name: str = "searx_search"
    description: str = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )
    wrapper: SearxSearchWrapper
    kwargs: dict = Field(default_factory=dict)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.wrapper.run(query, **self.kwargs)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return await self.wrapper.arun(query, **self.kwargs)


class SearxSearchResults(BaseTool):
    """Tool that queries a Searx instance and gets back json."""

    name: str = "Searx-Search-Results"
    description: str = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON array of the query results"
    )
    wrapper: SearxSearchWrapper
    num_results: int = 4
    kwargs: dict = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        extra = Extra.allow

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.wrapper.results(query, self.num_results, **self.kwargs))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return (
            await self.wrapper.aresults(query, self.num_results, **self.kwargs)
        ).__str__()
