"""Tool for the Wolfram Alpha API."""

from langchain.tools.base import BaseTool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


class WolframAlphaQueryRun(BaseTool):
    """Tool that adds the capability to query using the Wolfram Alpha SDK."""

    name = "query_wolfram_alpha"
    description = "Query Wolfram Alpha with the given query."
    api_wrapper: WolframAlphaAPIWrapper

    def func(self, query: str) -> str:
        """Use the WolframAlpha tool."""
        return self.api_wrapper.run(query)

    async def afunc(self, query: str) -> str:
        """Use the WolframAlpha tool asynchronously."""
        raise NotImplementedError("WolframAlphaQueryRun does not support async")
