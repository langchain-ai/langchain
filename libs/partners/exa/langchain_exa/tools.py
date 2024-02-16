"""Tool for the Exa Search API."""

from typing import Dict, List, Optional, Union

from exa_py import Exa  # type: ignore
from exa_py.api import HighlightsContentsOptions, TextContentsOptions  # type: ignore
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.tools import BaseTool

from langchain_exa._utilities import initialize_client


class ExaSearchResults(BaseTool):
    """Tool that queries the Metaphor Search API and gets back json."""

    name: str = "exa_search_results_json"
    description: str = (
        "A wrapper around Exa Search. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)
    exa_api_key: SecretStr = Field(default=None)

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        values = initialize_client(values)
        return values

    def _run(
        self,
        query: str,
        num_results: int,
        text_contents_options: Optional[Union[TextContentsOptions, bool]] = None,
        highlights: Optional[Union[HighlightsContentsOptions, bool]] = None,
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
            return self.client.search_and_contents(
                query,
                num_results=num_results,
                text=text_contents_options,  # type: ignore
                highlights=highlights,  # type: ignore
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                use_autoprompt=use_autoprompt,
            )  # type: ignore
        except Exception as e:
            return repr(e)


class ExaFindSimilarResults(BaseTool):
    """Tool that queries the Metaphor Search API and gets back json."""

    name: str = "exa_find_similar_results_json"
    description: str = (
        "A wrapper around Exa Find Similar. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)
    exa_api_key: SecretStr = Field(default=None)
    exa_base_url: Optional[str] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        values = initialize_client(values)
        return values

    def _run(
        self,
        url: str,
        num_results: int,
        text_contents_options: Optional[Union[TextContentsOptions, bool]] = None,
        highlights: Optional[Union[HighlightsContentsOptions, bool]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.client.find_similar_and_contents(
                url,
                num_results=num_results,
                text=text_contents_options,  # type: ignore
                highlights=highlights,  # type: ignore
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                exclude_source_domain=exclude_source_domain,
                category=category,
            )  # type: ignore
        except Exception as e:
            return repr(e)
