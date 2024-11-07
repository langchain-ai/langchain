"""Tool for the Metaphor search API."""

from typing import Dict, List, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from langchain_community.utilities.metaphor_search import MetaphorSearchAPIWrapper


@deprecated(
    since="0.0.15",
    removal="1.0",
    alternative="langchain_exa.ExaSearchResults",
)
class MetaphorSearchResults(BaseTool):  # type: ignore[override]
    """Tool that queries the Metaphor Search API and gets back json."""

    name: str = "metaphor_search_results_json"
    description: str = (
        "A wrapper around Metaphor Search. "
        "Input should be a Metaphor-optimized query. "
        "Output is a JSON array of the query results"
    )
    api_wrapper: MetaphorSearchAPIWrapper

    def _run(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(
                query,
                num_results,
                include_domains,
                exclude_domains,
                start_crawl_date,
                end_crawl_date,
                start_published_date,
                end_published_date,
                use_autoprompt,
            )
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.results_async(
                query,
                num_results,
                include_domains,
                exclude_domains,
                start_crawl_date,
                end_crawl_date,
                start_published_date,
                end_published_date,
                use_autoprompt,
            )
        except Exception as e:
            return repr(e)
