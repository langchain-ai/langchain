"""Tool for the Exa Search API."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from exa_py import Exa  # type: ignore[untyped-import]
from exa_py.api import (
    HighlightsContentsOptions,  # type: ignore[untyped-import]
    TextContentsOptions,  # type: ignore[untyped-import]
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_exa._utilities import initialize_client


class ExaSearchResults(BaseTool):  # type: ignore[override]
    r"""Exa Search tool.

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

            tool.invoke({"query": "what is the weather in SF", "num_results": 1})

        .. code-block:: python

            SearchResponse(
                results=[
                    Result(
                        url="https://www.wunderground.com/weather/37.8,-122.4",
                        id="https://www.wunderground.com/weather/37.8,-122.4",
                        title="San Francisco, CA Weather Conditionsstar_ratehome",
                        score=0.1843988299369812,
                        published_date="2023-02-23T01:17:06.594Z",
                        author=None,
                        text="The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.",
                        highlights=None,
                        highlight_scores=None,
                        summary=None,
                    )
                ],
                autoprompt_string=None,
            )

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke(
                {
                    "args": {"query": "what is the weather in SF", "num_results": 1},
                    "id": "1",
                    "name": tool.name,
                    "type": "tool_call",
                }
            )

        .. code-block:: python

            ToolMessage(
                content="Title: San Francisco, CA Weather Conditionsstar_ratehome\nURL: https://www.wunderground.com/weather/37.8,-122.4\nID: https://www.wunderground.com/weather/37.8,-122.4\nScore: 0.1843988299369812\nPublished Date: 2023-02-23T01:17:06.594Z\nAuthor: None\nText: The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.\nHighlights: None\nHighlight Scores: None\nSummary: None\n",
                name="exa_search_results_json",
                tool_call_id="1",
            )

    """  # noqa: E501

    name: str = "exa_search_results_json"
    description: str = (
        "A wrapper around Exa Search. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)
    exa_api_key: SecretStr = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        num_results: int = 10,
        text_contents_options: Optional[  # noqa: FBT001
            Union[TextContentsOptions, dict[str, Any], bool]
        ] = None,
        highlights: Optional[Union[HighlightsContentsOptions, bool]] = None,  # noqa: FBT001
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,  # noqa: FBT001
        livecrawl: Optional[Literal["always", "fallback", "never"]] = None,
        summary: Optional[Union[bool, dict[str, str]]] = None,  # noqa: FBT001
        type: Optional[Literal["neural", "keyword", "auto"]] = None,  # noqa: A002
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[list[dict], str]:
        # TODO: rename `type` to something else, as it is a reserved keyword
        """Use the tool.

        Args:
            query: The search query.
            num_results: The number of search results to return (1 to 100). Default: 10
            text_contents_options: How to set the page content of the results. Can be True or a dict with options like max_characters.
            highlights: Whether to include highlights in the results.
            include_domains: A list of domains to include in the search.
            exclude_domains: A list of domains to exclude from the search.
            start_crawl_date: The start date for the crawl (in YYYY-MM-DD format).
            end_crawl_date: The end date for the crawl (in YYYY-MM-DD format).
            start_published_date: The start date for when the document was published (in YYYY-MM-DD format).
            end_published_date: The end date for when the document was published (in YYYY-MM-DD format).
            use_autoprompt: Whether to use autoprompt for the search.
            livecrawl: Option to crawl live webpages if content is not in the index. Options: "always", "fallback", "never"
            summary: Whether to include a summary of the content. Can be a boolean or a dict with a custom query.
            type: The type of search, 'keyword', 'neural', or 'auto'.
            run_manager: The run manager for callbacks.

        """  # noqa: E501
        try:
            return self.client.search_and_contents(
                query,
                num_results=num_results,
                text=text_contents_options,
                highlights=highlights,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                use_autoprompt=use_autoprompt,
                livecrawl=livecrawl,
                summary=summary,
                type=type,
            )  # type: ignore[call-overload, misc]
        except Exception as e:
            return repr(e)


class ExaFindSimilarResults(BaseTool):  # type: ignore[override]
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

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        url: str,
        num_results: int = 10,
        text_contents_options: Optional[  # noqa: FBT001
            Union[TextContentsOptions, dict[str, Any], bool]
        ] = None,
        highlights: Optional[Union[HighlightsContentsOptions, bool]] = None,  # noqa: FBT001
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        exclude_source_domain: Optional[bool] = None,  # noqa: FBT001
        category: Optional[str] = None,
        livecrawl: Optional[Literal["always", "fallback", "never"]] = None,
        summary: Optional[Union[bool, dict[str, str]]] = None,  # noqa: FBT001
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[list[dict], str]:
        """Use the tool.

        Args:
            url: The URL to find similar pages for.
            num_results: The number of search results to return (1 to 100). Default: 10
            text_contents_options: How to set the page content of the results. Can be True or a dict with options like max_characters.
            highlights: Whether to include highlights in the results.
            include_domains: A list of domains to include in the search.
            exclude_domains: A list of domains to exclude from the search.
            start_crawl_date: The start date for the crawl (in YYYY-MM-DD format).
            end_crawl_date: The end date for the crawl (in YYYY-MM-DD format).
            start_published_date: The start date for when the document was published (in YYYY-MM-DD format).
            end_published_date: The end date for when the document was published (in YYYY-MM-DD format).
            exclude_source_domain: If `True`, exclude pages from the same domain as the source URL.
            category: Filter for similar pages by category.
            livecrawl: Option to crawl live webpages if content is not in the index. Options: "always", "fallback", "never"
            summary: Whether to include a summary of the content. Can be a boolean or a dict with a custom query.
            run_manager: The run manager for callbacks.

        """  # noqa: E501
        try:
            return self.client.find_similar_and_contents(
                url,
                num_results=num_results,
                text=text_contents_options,
                highlights=highlights,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                exclude_source_domain=exclude_source_domain,
                category=category,
                livecrawl=livecrawl,
                summary=summary,
            )  # type: ignore[call-overload, misc]
        except Exception as e:
            return repr(e)
