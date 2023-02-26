"""Tool for the Wolfram Alpha API."""

from langchain.tools.base import BaseTool
from langchain.utilities.arxiv_search import ArXivSearchAPIWrapper


class ArXivQueryRun(BaseTool):
    """Tool that adds the capability to query using the ArXiv Search API."""

    name = "ArXiv Search"
    description = (
        "A wrapper around ArXiv Search API. "
        "Useful for when you need to need information Scientific Papers."
        "Input should be a search query."
    )
    api_wrapper: ArXivSearchAPIWrapper

    def _run(self, query: str) -> str:
        """Use the ArXiv tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the ArXiv tool asynchronously."""
        raise NotImplementedError("ArXivQueryRun does not support async")
