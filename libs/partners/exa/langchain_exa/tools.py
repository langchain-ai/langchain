"""Tool for the Exa Search API."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Literal

from exa_py import Exa  # type: ignore[untyped-import]
from exa_py.api import (
    HighlightsContentsOptions,  # type: ignore[untyped-import]
    TextContentsOptions,  # type: ignore[untyped-import]
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from langchain_exa._utilities import (
    build_contents_options,
    initialize_client,
    warn_if_use_autoprompt,
)


class ExaSearchResults(BaseTool):  # type: ignore[override]
    r"""Exa Search tool.

    Setup:
        Install `langchain-exa` and set environment variable `EXA_API_KEY`.

        ```bash
        pip install -U langchain-exa
        export EXA_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_exa import ExaSearchResults

        tool = ExaSearchResults()
        ```

    Invocation with args:
        ```python
        tool.invoke({"query": "what is the weather in SF", "num_results": 1})
        ```

        ```python
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
        ```

    Invocation with ToolCall:

        ```python
        tool.invoke(
            {
                "args": {"query": "what is the weather in SF", "num_results": 1},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        ```python
        ToolMessage(
            content="Title: San Francisco, CA Weather Conditionsstar_ratehome\nURL: https://www.wunderground.com/weather/37.8,-122.4\nID: https://www.wunderground.com/weather/37.8,-122.4\nScore: 0.1843988299369812\nPublished Date: 2023-02-23T01:17:06.594Z\nAuthor: None\nText: The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.\nHighlights: None\nHighlight Scores: None\nSummary: None\n",
            name="exa_search_results_json",
            tool_call_id="1",
        )
        ```
    """  # noqa: E501

    name: str = "exa_search_results_json"
    description: str = (
        "Exa Search, one of the best web search APIs built for AI. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)  # type: ignore[assignment]
    async_client: Any = Field(default=None)
    exa_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        num_results: int = 10,
        text_contents_options: TextContentsOptions  # noqa: FBT001
        | dict[str, Any]
        | bool
        | None = None,
        highlights: HighlightsContentsOptions | bool | None = None,  # noqa: FBT001
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_crawl_date: str | None = None,
        end_crawl_date: str | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        use_autoprompt: bool | None = None,  # noqa: FBT001
        livecrawl: Literal["always", "fallback", "never"] | None = None,
        max_age_hours: int | None = None,
        summary: bool | dict[str, str] | None = None,  # noqa: FBT001
        type: Literal["auto", "deep", "fast"] = "auto",  # noqa: A002
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
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
            use_autoprompt: Deprecated and no longer sent to Exa; use `type="auto"`.
            livecrawl: Option to crawl live webpages if content is not in the index. Options: "always", "fallback", "never".
                Prefer `max_age_hours` for freshness.
            max_age_hours: The maximum age of cached content in hours. This is the recommended
                freshness control.
            summary: Whether to include a summary of the content. Can be a boolean or a dict with a custom query.
            type: The type of search, 'auto', 'deep', or 'fast'. Default: "auto".
            run_manager: The run manager for callbacks.

        """  # noqa: E501
        warn_if_use_autoprompt(use_autoprompt)
        try:
            return self.client.search(  # type: ignore[call-overload, return-value]
                query,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                contents=build_contents_options(  # type: ignore[arg-type]
                    text=text_contents_options,
                    highlights=highlights,
                    summary=summary,
                    livecrawl=livecrawl,
                    max_age_hours=max_age_hours,
                ),
                type=type,
            )  # type: ignore[call-overload, misc]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        num_results: int = 10,
        text_contents_options: TextContentsOptions  # noqa: FBT001
        | dict[str, Any]
        | bool
        | None = None,
        highlights: HighlightsContentsOptions | bool | None = None,  # noqa: FBT001
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_crawl_date: str | None = None,
        end_crawl_date: str | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        use_autoprompt: bool | None = None,  # noqa: FBT001
        livecrawl: Literal["always", "fallback", "never"] | None = None,
        max_age_hours: int | None = None,
        summary: bool | dict[str, str] | None = None,  # noqa: FBT001
        type: Literal["auto", "deep", "fast"] = "auto",  # noqa: A002
        run_manager: Any = None,
    ) -> Any:
        """Use the asynchronous Exa SDK."""
        warn_if_use_autoprompt(use_autoprompt)
        try:
            return await self.async_client.search(
                query,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                contents=build_contents_options(
                    text=text_contents_options,
                    highlights=highlights,
                    summary=summary,
                    livecrawl=livecrawl,
                    max_age_hours=max_age_hours,
                ),
                type=type,
            )
        except Exception as e:
            return repr(e)


class ExaFindSimilarResults(BaseTool):  # type: ignore[override]
    """Deprecated wrapper around Exa's ``/findSimilar`` endpoint.

    Exa has deprecated ``/findSimilar`` (and ``find_similar_and_contents`` in
    exa-py). Prefer :class:`ExaSearchResults` for new code. This tool remains
    for compatibility and emits a :class:`DeprecationWarning` when used.
    """

    name: str = "exa_find_similar_results_json"
    description: str = (
        "Deprecated. Prefer exa_search_results_json. "
        "Finds pages similar to a URL via Exa's deprecated /findSimilar endpoint."
    )
    client: Exa = Field(default=None)  # type: ignore[assignment]
    exa_api_key: SecretStr = Field(default=SecretStr(""))
    exa_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        url: str,
        num_results: int = 10,
        text_contents_options: TextContentsOptions  # noqa: FBT001
        | dict[str, Any]
        | bool
        | None = None,
        highlights: HighlightsContentsOptions | bool | None = None,  # noqa: FBT001
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_crawl_date: str | None = None,
        end_crawl_date: str | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        exclude_source_domain: bool | None = None,  # noqa: FBT001
        category: str | None = None,
        livecrawl: Literal["always", "fallback", "never"] | None = None,
        summary: bool | dict[str, str] | None = None,  # noqa: FBT001
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
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
        warnings.warn(
            "ExaFindSimilarResults is deprecated; Exa's /findSimilar endpoint is "
            "deprecated. Prefer ExaSearchResults.",
            DeprecationWarning,
            stacklevel=2,
        )
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


class ExaContentsInput(BaseModel):
    """Arguments for extracting contents from URLs."""

    urls: list[str] = Field(description="URLs to extract contents from.")
    text: bool | dict[str, Any] | None = None
    highlights: bool | dict[str, Any] | None = None
    summary: bool | dict[str, Any] | None = None
    max_age_hours: int | None = Field(default=None, ge=0)


class ExaAnswerInput(BaseModel):
    """Arguments for asking Exa a grounded question."""

    query: str
    text: bool | None = True
    output_schema: dict[str, Any] | None = None
    system_prompt: str | None = None
    user_location: str | None = None


class ExaAgentInput(BaseModel):
    """Arguments for running Exa's asynchronous Agent workflow."""

    query: str
    input: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    effort: Literal["low", "medium", "high", "xhigh", "auto"] | None = None
    previous_run_id: str | None = None
    data_sources: list[dict[str, Any]] | None = None
    timeout_ms: int = Field(default=3600000, ge=1)
    poll_interval: int = Field(default=1000, ge=1)


def _to_dict(value: Any) -> Any:
    """Normalize Exa SDK responses to plain snake_case dicts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_dict(value.model_dump(by_alias=False))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_dict(dataclasses.asdict(value))
    return value


class _ExaBaseTool(BaseTool):  # type: ignore[override]
    client: Exa = Field(default=None)  # type: ignore[assignment]
    async_client: Any = Field(default=None)
    exa_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        return initialize_client(values)


class ExaContents(_ExaBaseTool):
    """Extract text, highlights, or summaries from URLs with Exa.

    Returns a snake_case dict (``results``, ``statuses``, …).

    ```python
    from langchain_exa import ExaContents

    tool = ExaContents()
    result = tool.invoke({"urls": ["https://exa.ai"]})
    print(result["results"], result["statuses"])
    ```
    """

    name: str = "exa_contents"
    description: str = "Extract contents from one or more URLs using Exa."
    args_schema: type[BaseModel] = ExaContentsInput

    def _run(
        self,
        urls: list[str],
        *,
        text: bool | dict[str, Any] | None = None,
        highlights: bool | dict[str, Any] | None = None,
        summary: bool | dict[str, Any] | None = None,
        max_age_hours: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        options = build_contents_options(
            text=text,
            highlights=highlights,
            summary=summary,
            max_age_hours=max_age_hours,
        )
        return _to_dict(self.client.get_contents(urls, **options))

    async def _arun(
        self,
        urls: list[str],
        *,
        text: bool | dict[str, Any] | None = None,
        highlights: bool | dict[str, Any] | None = None,
        summary: bool | dict[str, Any] | None = None,
        max_age_hours: int | None = None,
        run_manager: Any = None,
    ) -> Any:
        client: Any = self.async_client or self.client
        options = build_contents_options(
            text=text,
            highlights=highlights,
            summary=summary,
            max_age_hours=max_age_hours,
        )
        result = await client.get_contents(urls, **options)
        return _to_dict(result)


class ExaAnswer(_ExaBaseTool):
    """Answer a question with grounded Exa citations.

    Returns a snake_case dict (``answer``, ``citations``, ``cost_dollars``).

    ```python
    from langchain_exa import ExaAnswer

    answer = ExaAnswer().invoke({"query": "What is Exa?"})
    print(answer["answer"], answer["citations"])
    ```
    """

    name: str = "exa_answer"
    description: str = (
        "Answer a question with a grounded answer and citations from Exa."
    )
    args_schema: type[BaseModel] = ExaAnswerInput

    def _run(
        self,
        query: str,
        *,
        text: bool | None = True,
        output_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_location: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        return _to_dict(
            self.client.answer(
                query,
                text=text,
                output_schema=output_schema,
                system_prompt=system_prompt,
                user_location=user_location,
            )
        )

    async def _arun(
        self,
        query: str,
        *,
        text: bool | None = True,
        output_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_location: str | None = None,
        run_manager: Any = None,
    ) -> Any:
        client: Any = self.async_client or self.client
        result = await client.answer(
            query,
            text=text,
            output_schema=output_schema,
            system_prompt=system_prompt,
            user_location=user_location,
        )
        return _to_dict(result)


class ExaAgent(_ExaBaseTool):
    """Run Exa's asynchronous multi-step Agent and poll to completion.

    Returns a snake_case dict (``status``, ``stop_reason``, ``cost_dollars``, …).

    ```python
    from langchain_exa import ExaAgent

    result = ExaAgent().invoke({"query": "Find the latest Exa product updates"})
    print(result["status"], result.get("output"))
    ```
    """

    name: str = "exa_agent"
    description: str = "Run Exa Agent for a long-running, multi-step grounded workflow."
    args_schema: type[BaseModel] = ExaAgentInput

    def _run(
        self,
        query: str,
        input: dict[str, Any] | None = None,  # noqa: A002
        *,
        output_schema: dict[str, Any] | None = None,
        effort: Literal["low", "medium", "high", "xhigh", "auto"] | None = None,
        previous_run_id: str | None = None,
        data_sources: list[dict[str, Any]] | None = None,
        timeout_ms: int = 3600000,
        poll_interval: int = 1000,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        create_kwargs: dict[str, Any] = {
            "query": query,
            "input": input,
            "output_schema": output_schema,
            "effort": effort,
            "previous_run_id": previous_run_id,
            "data_sources": data_sources,
        }
        run = self.client.agent.runs.create(**create_kwargs)
        result = self.client.agent.runs.poll_until_finished(
            run.id, timeout_ms=timeout_ms, poll_interval=poll_interval
        )
        return _to_dict(result)

    async def _arun(
        self,
        query: str,
        input: dict[str, Any] | None = None,  # noqa: A002
        *,
        output_schema: dict[str, Any] | None = None,
        effort: Literal["low", "medium", "high", "xhigh", "auto"] | None = None,
        previous_run_id: str | None = None,
        data_sources: list[dict[str, Any]] | None = None,
        timeout_ms: int = 3600000,
        poll_interval: int = 1000,
        run_manager: Any = None,
    ) -> Any:
        client: Any = self.async_client or self.client
        run = await client.agent.runs.create(
            query=query,
            input=input,
            output_schema=output_schema,
            effort=effort,
            previous_run_id=previous_run_id,
            data_sources=data_sources,
        )
        result = await client.agent.runs.poll_until_finished(
            run.id,
            timeout_ms=timeout_ms,
            poll_interval=poll_interval,
        )
        return _to_dict(result)
