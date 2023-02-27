"""Tool for the Google search API."""

from typing import List

from langchain.tools.base import BaseTool, BaseToolkit
from langchain.tools.google_search.tool import GoogleSearchRun
from langchain.utilities.google_search import GoogleSearchAPIWrapper


class GoogleSearchToolkit(BaseToolkit):
    """Tool that adds the capability to query the Google search API."""

    google_subscription_key: str
    google_search_url: str

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        wrapper = GoogleSearchAPIWrapper(
            google_api_key=self.google_subscription_key,
            google_search_url=self.google_search_url,
        )
        return [
            GoogleSearchRun(
                api_wrapper=wrapper,
            )
        ]
