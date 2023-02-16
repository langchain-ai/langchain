"""Tool for the Wolfram Alpha API."""

from langchain.tools.tool import Tool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


class WolframAlphaQueryRun(Tool):
    """Tool that adds the capability to query using the Wolfram Alpha SDK."""

    name = "query_wolfram_alpha"
    description = "Query Wolfram Alpha with the given query."
    api_wrapper: WolframAlphaAPIWrapper

    def func(self, query: str) -> str:
        """Use the WolframAlpha tool."""
        return self.api_wrapper.run(query)
