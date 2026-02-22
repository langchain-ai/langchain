from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator

from langchain_perplexity._utils import initialize_client


class PerplexitySearchRetriever(BaseRetriever):
    """Perplexity Search retriever."""

    k: int = Field(default=10, description="Max results (1-20)")
    max_tokens: int = Field(default=25000, description="Max tokens across all results")
    max_tokens_per_page: int = Field(default=1024, description="Max tokens per page")
    country: str | None = Field(default=None, description="ISO country code")
    search_domain_filter: list[str] | None = Field(
        default=None, description="Domain filter (max 20)"
    )
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None
    search_after_date: str | None = Field(
        default=None, description="Date filter (format: %m/%d/%Y)"
    )
    search_before_date: str | None = Field(
        default=None, description="Date filter (format: %m/%d/%Y)"
    )

    client: Any = Field(default=None, exclude=True)
    pplx_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        params = {
            "query": query,
            "max_results": self.k,
            "max_tokens": self.max_tokens,
            "max_tokens_per_page": self.max_tokens_per_page,
            "country": self.country,
            "search_domain_filter": self.search_domain_filter,
            "search_recency_filter": self.search_recency_filter,
            "search_after_date": self.search_after_date,
            "search_before_date": self.search_before_date,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = self.client.search.create(**params)

        return [
            Document(
                page_content=result.snippet,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "date": result.date,
                    "last_updated": result.last_updated,
                },
            )
            for result in response.results
        ]
