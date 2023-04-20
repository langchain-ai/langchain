"""Tool for the Arxiv API."""

from langchain.tools.base import BaseTool
from langchain.utilities.arxiv import ArxivAPIWrapper


class ArxivQueryRun(BaseTool):
    """Tool that adds the capability to search using the Arxiv API."""

    name = "Arxiv"
    description = (
        "A wrapper around Arxiv. "
        "Useful for getting summary of articles from arxiv.org. "
        "Input should be a search query."
    )
    api_wrapper: ArxivAPIWrapper

    def _run(self, query: str) -> str:
        """Use the Arxiv tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the Arxiv tool asynchronously."""
        raise NotImplementedError("ArxivAPIWrapper does not support async")
