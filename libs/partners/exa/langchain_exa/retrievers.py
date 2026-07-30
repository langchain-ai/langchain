"""Retriever using Exa Search API."""

from __future__ import annotations

from typing import Any, Literal

from exa_py import Exa  # type: ignore[untyped-import]
from exa_py.api import (
    HighlightsContentsOptions,  # type: ignore[untyped-import]
    TextContentsOptions,  # type: ignore[untyped-import]
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator

from langchain_exa._utilities import (
    build_contents_options,
    initialize_client,
    warn_if_use_autoprompt,
)


def _get_metadata(result: Any) -> dict[str, Any]:
    """Get the metadata from a result object."""
    metadata = {
        "title": result.title,
        "url": result.url,
        "id": result.id,
        "score": result.score,
        "published_date": result.published_date,
        "author": result.author,
    }
    if getattr(result, "highlights"):
        metadata["highlights"] = result.highlights
    if getattr(result, "highlight_scores"):
        metadata["highlight_scores"] = result.highlight_scores
    if getattr(result, "summary"):
        metadata["summary"] = result.summary
    return metadata


class ExaSearchRetriever(BaseRetriever):
    """Exa Search retriever."""

    k: int = 10  # num_results
    """The number of search results to return (1 to 100)."""
    include_domains: list[str] | None = None
    """A list of domains to include in the search."""
    exclude_domains: list[str] | None = None
    """A list of domains to exclude from the search."""
    start_crawl_date: str | None = None
    """The start date for the crawl (in YYYY-MM-DD format)."""
    end_crawl_date: str | None = None
    """The end date for the crawl (in YYYY-MM-DD format)."""
    start_published_date: str | None = None
    """The start date for when the document was published (in YYYY-MM-DD format)."""
    end_published_date: str | None = None
    """The end date for when the document was published (in YYYY-MM-DD format)."""
    use_autoprompt: bool | None = None
    """Deprecated and no longer sent to Exa; use `type="auto"` instead."""
    type: str = "auto"
    """The type of search, 'auto', 'deep', or 'fast'. Default: auto"""
    highlights: HighlightsContentsOptions | bool | None = None
    """Whether to include highlights of the results."""
    text_contents_options: (
        TextContentsOptions | dict[str, Any] | Literal[True] | None
    ) = None
    """How to set the page content of the results. Can be True or a dict with options
    like max_characters. Requested by default when no other content option is set."""
    livecrawl: Literal["always", "fallback", "never"] | None = None
    """Option to crawl live webpages if content is not in the index. Options: "always",
    "fallback", "never". Prefer `max_age_hours` for freshness."""
    max_age_hours: int | None = None
    """The maximum age of cached content in hours."""
    summary: bool | dict[str, str] | None = None
    """Whether to include a summary of the content. Can be a boolean or a dict with a
    custom query."""

    client: Exa = Field(default=None)  # type: ignore[assignment]
    async_client: Any = Field(default=None)
    exa_api_key: SecretStr = Field(default=SecretStr(""))
    exa_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _search_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments shared by the sync and async paths."""
        warn_if_use_autoprompt(self.use_autoprompt)
        return {
            "num_results": self.k,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "start_crawl_date": self.start_crawl_date,
            "end_crawl_date": self.end_crawl_date,
            "start_published_date": self.start_published_date,
            "end_published_date": self.end_published_date,
            "contents": build_contents_options(
                text=self.text_contents_options,
                highlights=self.highlights,
                summary=self.summary,
                livecrawl=self.livecrawl,
                max_age_hours=self.max_age_hours,
            ),
            "type": self.type,
        }

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        response = self.client.search(  # type: ignore[call-overload]
            query,
            **self._search_kwargs(),
        )  # type: ignore[call-overload, misc]

        return [
            Document(
                page_content=(result.text or ""),
                metadata=_get_metadata(result),
            )
            for result in response.results
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        """Use the asynchronous Exa SDK."""
        response = await self.async_client.search(query, **self._search_kwargs())
        return [
            Document(
                page_content=(result.text or ""),
                metadata=_get_metadata(result),
            )
            for result in response.results
        ]
