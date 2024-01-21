"""Tool for the Metaphor search API."""

from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.metaphor_search import MetaphorSearchAPIWrapper


class MetaphorSearchResultsToolInput(BaseModel):
    query: str = Field(description="Metaphor-optimized query")
    num_results: int = Field(
        description="Number of results that needs to be returned from the query"
    )
    include_domains: Optional[List[str]] = Field(description="Domains Included")
    exclude_domains: Optional[List[str]] = Field(description="Domains Excluded")
    start_crawl_date: Optional[str] = Field(description="Crawling Start Date")
    end_crawl_date: Optional[str] = Field(description="Crawling End Date")
    start_published_date: Optional[str] = Field(description="Pubish Start Date")
    end_published_date: Optional[str] = Field(description="Publish End Date")
    use_autoprompt: Optional[bool] = Field(description="Whether to use autoprompt")


class MetaphorSearchResults(BaseTool):
    """Tool that queries the Metaphor Search API and gets back json."""

    name: str = "metaphor_search_results_json"
    description: str = (
        "A wrapper around Metaphor Search. "
        "Input should be a Metaphor-optimized query. "
        "Output is a JSON array of the query results"
    )
    api_wrapper: MetaphorSearchAPIWrapper
    args_schema: Type[MetaphorSearchResultsToolInput]

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
