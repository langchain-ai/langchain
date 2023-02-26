"""Tool for the Bing search API."""

from typing import List

from langchain.tools.arxiv_search.tool import ArXivQueryRun
from langchain.tools.base import BaseTool, BaseToolkit
from langchain.utilities.arxiv_search import ArXivSearchAPIWrapper


class ArXivSearchToolkit(BaseToolkit):
    """Tool that adds the capability to query the Bing search API."""

    max_results: int
    sort_by: str
    sort_order: str

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        wrapper = ArXivSearchAPIWrapper(
            max_results=self.max_results,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
        )
        return [
            ArXivQueryRun(
                api_wrapper=wrapper,
            )
        ]
