"""Tool for the Bing search API."""

from langchain.tools.tool import Tool
from langchain.utilities.bing_search import BingSearchAPIWrapper


class BingSearchRun(Tool):
    """Tool that adds the capability to query the Bing search API."""

    name = "bing_search"
    description = "Execute the Bing search API."
    api_wrapper: BingSearchAPIWrapper

    def func(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
