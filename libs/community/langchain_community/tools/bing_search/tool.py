"""Tool for the Bing search API."""

from typing import Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.bing_search import BingSearchAPIWrapper


class BingSearchRun(BaseTool):
    """Tool that queries the Bing search API."""

    name: str = "bing_search"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: BingSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class BingSearchResults(BaseTool):
    """Tool that queries the Bing Search API and gets a list of results."""

    name: str = "bing_search_results_json"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is an array of the query results."
    )
    num_results: int = 4
    """Max search results to return, default is 4."""
    api_wrapper: BingSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(query, self.num_results)
        except Exception as e:
            return repr(e)
