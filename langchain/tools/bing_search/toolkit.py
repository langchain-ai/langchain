"""Tool for the Bing search API."""

from typing import List

from langchain.tools.bing_search.tool import BingSearchRun
from langchain.tools.tool import Tool
from langchain.tools.toolkit import Toolkit
from langchain.utilities.bing_search import BingSearchAPIWrapper


class BingSearchToolkit(Toolkit):
    """Tool that adds the capability to query the Bing search API."""

    bing_subscription_key: str
    bing_search_url: str

    def get_tools(self) -> List[Tool]:
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
