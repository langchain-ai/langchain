"""Tool for the Bing search API."""

from typing import List

from langchain.tools.base import BaseTool, BaseToolkit
from langchain.tools.bing_search.tool import BingSearchRun
from langchain.utilities.bing_search import BingSearchAPIWrapper


class BingSearchToolkit(BaseToolkit):
    """Tool that adds the capability to query the Bing search API."""

    bing_subscription_key: str
    bing_search_url: str

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        wrapper = BingSearchAPIWrapper(
            bing_subscription_key=self.bing_subscription_key,
            bing_search_url=self.bing_search_url,
        )
        return [
            BingSearchRun(
                api_wrapper=wrapper,
            )
        ]
