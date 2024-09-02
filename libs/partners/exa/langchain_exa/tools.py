"""Tool for the Exa Search API."""  # type: ignore[import-not-found, import-not-found]

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
    """Exa Search tool.

    Setup:
        Install ``langchain-exa`` and set environment variable ``EXA_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-exa
            export EXA_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain-exa import ExaSearchResults

            tool = ExaSearchResults()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query":"what is the weather in SF","num_results":1})

        .. code-block:: python

            SearchResponse(results=[Result(url='https://www.wunderground.com/weather/37.8,-122.4', id='https://www.wunderground.com/weather/37.8,-122.4', title='San Francisco, CA Weather Conditionsstar_ratehome', score=0.1843988299369812, published_date='2023-02-23T01:17:06.594Z', author=None, text='The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.', highlights=None, highlight_scores=None, summary=None)], autoprompt_string=None)

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke({"args": {"query":"what is the weather in SF","num_results":1}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            ToolMessage(content='Title: San Francisco, CA Weather Conditionsstar_ratehome\nURL: https://www.wunderground.com/weather/37.8,-122.4\nID: https://www.wunderground.com/weather/37.8,-122.4\nScore: 0.1843988299369812\nPublished Date: 2023-02-23T01:17:06.594Z\nAuthor: None\nText: The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.\nHighlights: None\nHighlight Scores: None\nSummary: None\n', name='exa_search_results_json', tool_call_id='1')

    """  # noqa: E501

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
