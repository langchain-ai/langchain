"""Tool for the Google search API."""

from langchain.tools.base import BaseTool
from langchain.utilities.google_search import GoogleSearchAPIWrapper


class GoogleSearchRun(BaseTool):
    """Tool that adds the capability to query the Google search API."""

    name = "google_search"
    description = "Execute the Google search API."
    api_wrapper: GoogleSearchAPIWrapper

    def func(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def afunc(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleSearchRun does not support async")
