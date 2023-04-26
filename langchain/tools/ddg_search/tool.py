"""Tool for the DuckDuckGo search API."""

from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


class DuckDuckGoSearchTool(BaseTool):
    """Tool that adds the capability to query the DuckDuckGo search API."""

    name = "DuckDuckGo Search"
    description = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DuckDuckGoSearch does not support async")
