"""Tool for the SearxNG search API."""
from pydantic import Extra

from langchain.tools.base import BaseTool
from langchain.utilities.searx_search import SearxSearchWrapper


class SearxSearchRun(BaseTool):
    """Tool that adds the capability to query a Searx instance."""

    name = "Searx Search"
    description = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query."
    )
    wrapper: SearxSearchWrapper

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        return await self.wrapper.arun(query)


class SearxSearchResults(BaseTool):
    """Tool that has capability to query a Searx instance and get back json."""

    name = "Searx Search"
    description = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON array of the query results"
    )
    wrapper: SearxSearchWrapper
    num_results: int = 4

    class Config:
        """Pydantic config."""

        extra = Extra.allow

    def _run(self, query: str) -> str:
        """Use the tool."""
        return str(self.wrapper.results(query, self.num_results))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        return (await self.wrapper.aresults(query, self.num_results)).__str__()
