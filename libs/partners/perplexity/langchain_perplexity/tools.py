from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_perplexity._utils import initialize_client


class PerplexitySearchResults(BaseTool):
    """Perplexity Search tool."""

    name: str = "perplexity_search_results_json"
    description: str = (
        "A wrapper around Perplexity Search. "
        "Input should be a search query. "
        "Output is a JSON array of the query results"
    )
    client: Any = Field(default=None, exclude=True)
    pplx_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str | list[str],
        max_results: int = 10,
        country: str | None = None,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: Literal["day", "week", "month", "year"] | None = None,
        search_after_date: str | None = None,
        search_before_date: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
        """Use the tool."""
        try:
            params = {
                "query": query,
                "max_results": max_results,
                "country": country,
                "search_domain_filter": search_domain_filter,
                "search_recency_filter": search_recency_filter,
                "search_after_date": search_after_date,
                "search_before_date": search_before_date,
            }
            params = {k: v for k, v in params.items() if v is not None}
            response = self.client.search.create(**params)
            return [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "date": result.date,
                    "last_updated": result.last_updated,
                }
                for result in response.results
            ]
        except Exception as e:
            msg = f"Perplexity search failed: {type(e).__name__}"
            return msg
